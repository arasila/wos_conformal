// vy.cu
// Recommended build (RTX 4090): nvcc -O3 -arch=sm_89 --use_fast_math vy.cu -o vy -lcurand
// Run:   ./vy

#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <math.h>
#include <time.h>
#include <string.h>

#include <cuda_runtime.h>
#include <curand_kernel.h>

#define PI 3.14159265358979323846
#define x_center 0.3 
#define y_center 0.5 
#define STEP 0.1  //Step size of the five-point formula. To be passed as in function dFivePoint as h
#define ANGLE (PI / 3) 
#define NUM_POINTS 1000 //Number of node in each list of fm2h, fmh, fph, fp2h
#define NUM_SHOTS 1.0E6 /* Total number of shots for each of fm2h, fmh, fph, fp2h. Must be a float number */
#define EPSILON 1e-8
//#define TEST
#define length(x, y) sqrt((x)*(x)+(y)*(y))

#define v1_x 0.8660254037844386
#define v1_y 0.5
#define v2_x 0.0
#define v2_y 1.0
#define v3_x -0.7071067811865475
#define v3_y 0.7071067811865475
#define v4_x 1.0
#define v4_y 0.0

inline unsigned int walk_on_spheres(double x, double y, unsigned int num_shots, double epsilon); 
inline double dFivePoint(double fm2h, double fmh, double fph, double fp2h, double h);
double calculateAverage(unsigned int array[], int size);
inline void print_fm2h_results(unsigned int* list, int num_of_pts);
inline void print_fmh_results(unsigned int* list, int num_of_pts);
inline void print_fph_results(unsigned int* list, int num_of_pts);
inline void print_fp2h_results(unsigned int* list, int num_of_pts);

/* ---------------- CUDA error check ---------------- */
#define CHECK_CUDA(call)                                                      \
    do {                                                                      \
        cudaError_t err__ = (call);                                           \
        if (err__ != cudaSuccess) {                                           \
            fprintf(stderr, "CUDA error %s at %s:%d\n",                       \
                    cudaGetErrorString(err__), __FILE__, __LINE__);           \
            exit(EXIT_FAILURE);                                               \
        }                                                                     \
    } while (0)

__device__ unsigned int walk_on_spheres_dev(double x1, double y1,
                                            unsigned int NN, double eps,
                                            curandStatePhilox4_32_10_t* rng)
{
    double wx = x1, wy = y1;
    unsigned int cnt = 0;
    int j = 0;
    double distance, theta, denom;

    while (j < (int)NN) {
        denom = wx*wx + wy*wy;
        if (denom > 1.0) {
            wx = wx/denom;
            wy = wy/denom;
        }

        if (wy >= 0 &&
            !(((tan(1.0 / 6.0 * PI)*wx < wy && wx > 0) || wx*tan(3.0 / 4.0 * PI)>wy))) {
            distance = sqrt(fmin(
                (wx - 1) * (wx - 1) + wy * wy,
                fmin(
                    (wx - v1_x) * (wx - v1_x) + (wy - v1_y) * (wy - v1_y),
                    fmin(
                        (wx - v2_x) * (wx - v2_x) + (wy - v2_y) * (wy - v2_y),
                        (wx - v3_x) * (wx - v3_x) + (wy - v3_y) * (wy - v3_y)
                    )
                )
            ));
        } else {
            distance = 1 - sqrt(denom);
        }

        if (distance < eps) {
            cnt += (wx > -0.1 && wy > 0.1) ? 1u : 0u;
            wx = x1;
            wy = y1;
            j++;
        } else {
            double u = curand_uniform_double(rng);   // (0,1]
            theta = 2.0 * PI * u;
            wx += distance * cos(theta);
            wy += distance * sin(theta);
        }
    }
    return cnt;
}

__device__ unsigned int walk_on_spheres_strided(double x1,
                                                double y1,
                                                unsigned int NN,
                                                double eps,
                                                curandStatePhilox4_32_10_t* rng,
                                                unsigned int j_start,
                                                unsigned int j_stride)
{
    double wx = x1, wy = y1;
    unsigned int cnt = 0;
    unsigned int j   = j_start;
    double distance, theta, denom;

    while (j < NN) {
        denom = wx*wx + wy*wy;
        if (denom > 1.0) {
            wx = wx/denom;
            wy = wy/denom;
        }

        if (wy >= 0 &&
            !(((tan(1.0 / 6.0 * PI)*wx < wy && wx > 0) || wx*tan(3.0 / 4.0 * PI)>wy))) {
            distance = sqrt(fmin(
                (wx - 1) * (wx - 1) + wy * wy,
                fmin(
                    (wx - v1_x) * (wx - v1_x) + (wy - v1_y) * (wy - v1_y),
                    fmin(
                        (wx - v2_x) * (wx - v2_x) + (wy - v2_y) * (wy - v2_y),
                        (wx - v3_x) * (wx - v3_x) + (wy - v3_y) * (wy - v3_y)
                    )
                )
            ));
        } else {
            distance = 1 - sqrt(denom);
        }

        if (distance < eps) {
            // Keep hit logic identical to walk_on_spheres_dev: wx > -0.1 && wy > 0.1
            cnt += (wx > -0.1 && wy > 0.1) ? 1u : 0u;
            wx = x1;
            wy = y1;
            j += j_stride;
        } else {
            double u = curand_uniform_double(rng);   // (0,1]
            theta = 2.0 * PI * u;
            wx += distance * cos(theta);
            wy += distance * sin(theta);
        }
    }
    return cnt;
}

__global__ void vy_kernel(int n,
                          double x1, double y1,
                          double x2, double y2,
                          double x3, double y3,
                          double x4, double y4,
                          unsigned int num_shots,
                          double eps,
                          unsigned int* fm2h,
                          unsigned int* fmh,
                          unsigned int* fph,
                          unsigned int* fp2h,
                          unsigned long long seed)
{
    int p = blockIdx.x; 
    if (p >= n) return;

    int tid = threadIdx.x;

    curandStatePhilox4_32_10_t rng;
    curand_init(seed, (unsigned long long)p * blockDim.x + tid, 0ULL, &rng);

    unsigned int local_m2 = walk_on_spheres_strided(x1, y1, num_shots, eps,
                                                    &rng,
                                                    (unsigned int)tid,
                                                    (unsigned int)blockDim.x);
    unsigned int local_m1 = walk_on_spheres_strided(x2, y2, num_shots, eps,
                                                    &rng,
                                                    (unsigned int)tid,
                                                    (unsigned int)blockDim.x);
    unsigned int local_p1 = walk_on_spheres_strided(x3, y3, num_shots, eps,
                                                    &rng,
                                                    (unsigned int)tid,
                                                    (unsigned int)blockDim.x);
    unsigned int local_p2 = walk_on_spheres_strided(x4, y4, num_shots, eps,
                                                    &rng,
                                                    (unsigned int)tid,
                                                    (unsigned int)blockDim.x);

    extern __shared__ unsigned int sdata[];
    unsigned int* s_m2 = sdata;
    unsigned int* s_m1 = sdata + blockDim.x;
    unsigned int* s_p1 = sdata + 2 * blockDim.x;
    unsigned int* s_p2 = sdata + 3 * blockDim.x;

    s_m2[tid] = local_m2;
    s_m1[tid] = local_m1;
    s_p1[tid] = local_p1;
    s_p2[tid] = local_p2;
    __syncthreads();

    for (int offset = blockDim.x >> 1; offset > 0; offset >>= 1) {
        if (tid < offset) {
            s_m2[tid] += s_m2[tid + offset];
            s_m1[tid] += s_m1[tid + offset];
            s_p1[tid] += s_p1[tid + offset];
            s_p2[tid] += s_p2[tid + offset];
        }
        __syncthreads();
    }

    if (tid == 0) {
        fm2h[p] = s_m2[0];
        fmh[p]  = s_m1[0];
        fph[p]  = s_p1[0];
        fp2h[p] = s_p2[0];
    }
}

int main() {
    time_t rawtime;
    struct tm * timeinfo;
    time( &rawtime );
    timeinfo = localtime( &rawtime );
    printf( "Current local time and date: %s", asctime (timeinfo) );
    printf( "Every point has %lf shots\n", NUM_SHOTS);

    unsigned int fm2h_list[NUM_POINTS] = {0};
    unsigned int fmh_list[NUM_POINTS]  = {0};
    unsigned int fph_list[NUM_POINTS]  = {0};
    unsigned int fp2h_list[NUM_POINTS] = {0};

    srand(time(NULL));

    double x1 = x_center + (-2) * STEP * cos(ANGLE) ;
    double y1 = y_center + (-2) * STEP * sin(ANGLE) ;
    double x2 = x_center + (-1) * STEP * cos(ANGLE) ;
    double y2 = y_center + (-1) * STEP * sin(ANGLE) ;
    double x3 = x_center + 1 * STEP * cos(ANGLE) ;
    double y3 = y_center + 1 * STEP * sin(ANGLE) ;
    double x4 = x_center + 2 * STEP * cos(ANGLE) ;
    double y4 = y_center + 2 * STEP * sin(ANGLE) ;

    int N = NUM_POINTS;

    const unsigned int shots = (unsigned int)NUM_SHOTS;

    unsigned int *d_fm2h = NULL, *d_fmh = NULL;
    unsigned int *d_fph  = NULL, *d_fp2h = NULL;

    CHECK_CUDA(cudaMalloc((void**)&d_fm2h, N * sizeof(unsigned int)));
    CHECK_CUDA(cudaMalloc((void**)&d_fmh,  N * sizeof(unsigned int)));
    CHECK_CUDA(cudaMalloc((void**)&d_fph,  N * sizeof(unsigned int)));
    CHECK_CUDA(cudaMalloc((void**)&d_fp2h, N * sizeof(unsigned int)));

    const int TPB    = 256;
    const int blocks = N;
    unsigned long long seed = (unsigned long long)time(NULL);

    vy_kernel<<<blocks, TPB, TPB * 4 * sizeof(unsigned int)>>>(N,
                               x1, y1,
                               x2, y2,
                               x3, y3,
                               x4, y4,
                               shots,
                               EPSILON,
                               d_fm2h, d_fmh, d_fph, d_fp2h,
                               seed);
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaMemcpy(fm2h_list, d_fm2h, N * sizeof(unsigned int), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(fmh_list,  d_fmh,  N * sizeof(unsigned int), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(fph_list,  d_fph,  N * sizeof(unsigned int), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(fp2h_list, d_fp2h, N * sizeof(unsigned int), cudaMemcpyDeviceToHost));

    print_fm2h_results(fm2h_list, N);
    print_fmh_results (fmh_list,  N);
    print_fph_results (fph_list,  N);
    print_fp2h_results(fp2h_list, N);

    CHECK_CUDA(cudaFree(d_fm2h));
    CHECK_CUDA(cudaFree(d_fmh));
    CHECK_CUDA(cudaFree(d_fph));
    CHECK_CUDA(cudaFree(d_fp2h));

    time( &rawtime );
    timeinfo = localtime( &rawtime );
    return 0;
}


inline unsigned int walk_on_spheres(double x1, double y1, unsigned int NN, double eps) {
    double wx = x1, wy = y1;
    unsigned int cnt = 0;
    int j = 0;
    double distance, theta, denom;
    while (j < NN) {
        denom = wx*wx + wy*wy;
        if (denom > 1) {
                wx = wx/denom;
                wy = wy/denom;
        }

        if (wy >= 0 && !(((tan(1.0 / 6.0 * PI)*wx < wy && wx > 0) || wx*tan(3.0 / 4.0 * PI)>wy))){
            distance = sqrt(fmin(
                (wx - 1) * (wx - 1) + wy * wy,
                fmin(
                    (wx - v1_x) * (wx - v1_x) + (wy - v1_y) * (wy - v1_y),
                    fmin(
                        (wx - v2_x) * (wx - v2_x) + (wy - v2_y) * (wy - v2_y),
                        (wx - v3_x) * (wx - v3_x) + (wy - v3_y) * (wy - v3_y)
                    )
                )
            ));
        } else {
            distance = 1 - sqrt(denom);
        }
        if (distance < eps) {
                cnt += (wx > -0.1 && wy > 0.1) ? 1 : 0;
                wx = x1;
                wy = y1;
                j++;
        } else {
                theta = ((double)(rand()))*2*PI/(RAND_MAX);
                wx += distance * cos(theta);
                wy += distance * sin(theta);
        }
    }
    return cnt;
}

inline double dFivePoint(double fm2h, double fmh, double fph, double fp2h, double h){
    return (fm2h - 8*fmh + 8*fph - fp2h)/(12 * h);	
}

// Function to calculate the average of an array
double calculateAverage(unsigned int array[], int size) {
    int sum = 0;

    for (int i = 0; i < size; i++) {
        sum += array[i];
    }

    return (double)sum / size;
}

inline void print_fm2h_results(unsigned int* list, int num_of_pts) {
    FILE* fd = fopen("cfm2h.txt", "w");
    for (int i = 0; i < num_of_pts; i++) {	
        fprintf(fd, "%u\n", list[i] );
    }
    fclose(fd);
}
inline void print_fmh_results(unsigned int* list, int num_of_pts) {
    FILE* fd = fopen("cfmh.txt", "w");
    for (int i = 0; i < num_of_pts; i++) {	
        fprintf(fd, "%u\n", list[i] );
    }
    fclose(fd);
}
inline void print_fph_results(unsigned int* list, int num_of_pts) {
    FILE* fd = fopen("cfph.txt", "w");
    for (int i = 0; i < num_of_pts; i++) {	
        fprintf(fd, "%u\n", list[i] );
    }
    fclose(fd);
}
inline void print_fp2h_results(unsigned int* list, int num_of_pts) {
    FILE* fd = fopen("cfp2h.txt", "w");
    for (int i = 0; i < num_of_pts; i++) {	
        fprintf(fd, "%u\n", list[i] );
    }
    fclose(fd);
}
