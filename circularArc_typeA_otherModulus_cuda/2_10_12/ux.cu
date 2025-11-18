// ux_cuda.cu
// CUDA version of othermodule_circulararc ux.c
// Build: nvcc -O3 -arch=sm_89 ux.cu -o ux -lcurand
// Run:   ./ux

#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <math.h>
#include <time.h>
#include <string.h>

#include <cuda_runtime.h>
#include <curand_kernel.h>

/*
typedef struct {
    double x;
    double y;
    double result;
    unsigned int num_shots;
} point_t, *Point;
*/

#define HEIGHT 1.0
#define PI 3.14159265358979323846
#define x_center 0.5 
#define y_center 0.5 
#define STEP 0.1  // Step size of the five-point formula
#define ANGLE (- PI / 4) 
#define NUM_POINTS 1000
#define NUM_SHOTS 1.0E6 /* Must be a float number */
#define EPSILON 1e-7
#define length(x, y) sqrt((x)*(x)+(y)*(y))

#define center2x 0.13165249758739628
#define center2y 1.0
#define r2       0.13165249758739545
#define center4x 1.0
#define center4y 0.13165249758739628
#define r4       0.13165249758739628
#define Tarc2Centerx 0.9899038686932885
#define Tarc2Centery 0.14174862889406872
#define Tarc2Radius  0.0015307015898607065
#define Tarc4Centerx 0.14174862889407122
#define Tarc4Centery 0.9899038686932877
#define Tarc4Radius  0.0015307015898573843

// ---------- CUDA error check ----------
#define CHECK_CUDA(call)                                                      \
    do {                                                                      \
        cudaError_t err__ = (call);                                           \
        if (err__ != cudaSuccess) {                                           \
            fprintf(stderr, "CUDA error %s at %s:%d\n",                       \
                    cudaGetErrorString(err__), __FILE__, __LINE__);           \
            exit(EXIT_FAILURE);                                               \
        }                                                                     \
    } while (0)

inline unsigned int walk_on_spheres(double x, double y, unsigned int num_shots, double epsilon); 
inline double dFivePoint(double fm2h, double fmh, double fph, double fp2h, double h);
double calculateAverage(unsigned int array[], int size);
inline void print_fm2h_results(unsigned int* list, int num_of_pts);
inline void print_fmh_results(unsigned int* list, int num_of_pts);
inline void print_fph_results(unsigned int* list, int num_of_pts);
inline void print_fp2h_results(unsigned int* list, int num_of_pts);

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
    unsigned int j = j_start;
    double distance, theta, denom;

    while (j < NN) {
        denom = (wx*wx + wy*wy);
        if (denom > 1.0) {
            wx = wx / denom;
            wy = wy / denom;
        }

        double dist1 = length(wx - center2x, wy - center2y) - r2;
        double dist2 = length(wx - center4x, wy - center4y) - r4;
        distance = (dist1 > dist2) ? dist2 : dist1;

        if (distance < eps) {
            cnt += (wx < 0.5) ? 0u : 1u;
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

__device__ unsigned int walk_on_spheres_dev(double x1,
                                            double y1,
                                            unsigned int NN,
                                            double eps,
                                            curandStatePhilox4_32_10_t* rng)
{
    double wx = x1, wy = y1;
    unsigned int cnt = 0;
    unsigned int j = 0;
    double distance, theta, denom;

    while (j < NN) {
        denom = (wx*wx + wy*wy);
        if (denom > 1.0) {
            wx = wx / denom;
            wy = wy / denom;
        }

        double dist1 = length(wx - center2x, wy - center2y) - r2;
        double dist2 = length(wx - center4x, wy - center4y) - r4;
        distance = (dist1 > dist2) ? dist2 : dist1;

        if (distance < eps) {
            cnt += (wx < 0.5) ? 0u : 1u;
            wx = x1;
            wy = y1;
            ++j;
        } else {
            double u = curand_uniform_double(rng);   // (0,1]
            theta = 2.0 * PI * u;
            wx += distance * cos(theta);
            wy += distance * sin(theta);
        }
    }

    return cnt;
}

__global__ void ux_kernel(int n,
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
    time(&rawtime);
    timeinfo = localtime(&rawtime);
    printf("Current local time and date: %s", asctime(timeinfo));
    printf("Every point has %lf shots\n", NUM_SHOTS);

    unsigned int fm2h_list[NUM_POINTS] = {0};
    unsigned int fmh_list[NUM_POINTS]  = {0};
    unsigned int fph_list[NUM_POINTS]  = {0};
    unsigned int fp2h_list[NUM_POINTS] = {0};

    srand((unsigned)time(NULL)); 
    int N = NUM_POINTS;

    double x1 = x_center + (-2) * STEP * cos(ANGLE);
    double y1 = y_center + (-2) * STEP * sin(ANGLE);
    double x2 = x_center + (-1) * STEP * cos(ANGLE);
    double y2 = y_center + (-1) * STEP * sin(ANGLE);
    double x3 = x_center + ( 1) * STEP * cos(ANGLE);
    double y3 = y_center + ( 1) * STEP * sin(ANGLE);
    double x4 = x_center + ( 2) * STEP * cos(ANGLE);
    double y4 = y_center + ( 2) * STEP * sin(ANGLE);

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
    size_t shmem_size = 4 * TPB * sizeof(unsigned int);  // m2, m1, p1, p2

    ux_kernel<<<blocks, TPB, shmem_size>>>(N,
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

    return 0;
}


inline unsigned int walk_on_spheres(double x1, double y1, unsigned int NN, double eps) {
    double wx = x1, wy = y1;
    unsigned int cnt = 0;
    int j = 0;
    double distance, theta, denom;

    while (j < NN) {
        denom = (wx*wx + wy*wy);
        if (denom > 1) {
            wx = wx/denom;
            wy = wy/denom;
        }
        double dist1 = length( wx-center2x, wy-center2y ) - r2;
        double dist2 = length( wx-center4x, wy-center4y ) - r4;
        distance = dist1 > dist2 ? dist2 : dist1;
        if (distance < eps) {
            cnt += (wx < 0.5) ? 0 : 1;
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

double calculateAverage(unsigned int array[], int size) {
    int sum = 0;
    for (int i = 0; i < size; i++) {
        sum += array[i];
    }
    return (double)sum / size;
}

inline void print_fm2h_results(unsigned int* list, int num_of_pts) {
    FILE* fd = fopen("fm2h.txt", "w");
    if (!fd) { perror("fopen fm2h"); return; }
    for (int i = 0; i < num_of_pts; i++) {
        fprintf(fd, "%u\n", *(list+i) );
    }
    fclose(fd);
}
inline void print_fmh_results(unsigned int* list, int num_of_pts) {
    FILE* fd = fopen("fmh.txt", "w");
    if (!fd) { perror("fopen fmh"); return; }
    for (int i = 0; i < num_of_pts; i++) {
        fprintf(fd, "%u\n", *(list+i) );
    }
    fclose(fd);
}
inline void print_fph_results(unsigned int* list, int num_of_pts) {
    FILE* fd = fopen("fph.txt", "w");
    if (!fd) { perror("fopen fph"); return; }
    for (int i = 0; i < num_of_pts; i++) {
        fprintf(fd, "%u\n", *(list+i) );
    }
    fclose(fd);
}
inline void print_fp2h_results(unsigned int* list, int num_of_pts) {
    FILE* fd = fopen("fp2h.txt", "w");
    if (!fd) { perror("fopen fp2h"); return; }
    for (int i = 0; i < num_of_pts; i++) {
        fprintf(fd, "%u\n", *(list+i) );
    }
    fclose(fd);
}
