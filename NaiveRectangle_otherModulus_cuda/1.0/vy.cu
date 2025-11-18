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
#define y_center  (HEIGHT / 2)  
#define STEP 0.1  //Step size of the five-point formula. To be passed as in function dFivePoint as h
#define ANGLE ( PI / 2 )
#define NUM_POINTS 1000 //Number of node in each list of fm2h, fmh, fph, fp2h
#define NUM_SHOTS 1.0E6 /* Total number of shots for each of fm2h, fmh, fph, fp2h. Must be a float number */
#define EPSILON 1e-7
//#define TEST
#define length(x, y) sqrt((x)*(x)+(y)*(y))

inline unsigned int walk_on_spheres(double x, double y, unsigned int num_shots, double epsilon); 
inline double dFivePoint(double fm2h, double fmh, double fph, double fp2h, double h);
double calculateAverage(unsigned int array[], int size);
inline void print_fm2h_results(unsigned int* list, int num_of_pts);
inline void print_fmh_results(unsigned int* list, int num_of_pts);
inline void print_fph_results(unsigned int* list, int num_of_pts);
inline void print_fp2h_results(unsigned int* list, int num_of_pts);

/* ---------------- device version of walk_on_spheres ---------------- */

__device__ unsigned int walk_on_spheres_device(double x1, double y1,
                                               unsigned int num_shots,
                                               double epsilon,
                                               curandStatePhilox4_32_10_t *state){
    double wx = x1;
    double wy = y1;
    unsigned int mark = 0;
    unsigned int j = 0;
    while (j < num_shots) {
        if (wx < 0.0) {
            wx = -wx;
        } else if (wx > 1.0){
            wx = 2.0 - wx;
        }
        double r = (HEIGHT - wy) > wy ? wy : (HEIGHT - wy);
        if (r < EPSILON){
            if (wy > HEIGHT/2.0) mark++;
            wx = x1;
            wy = y1;
            j++;
            continue;
        }
        double u1 = curand_uniform_double(state) * 2.0 * PI;
        double z1 = sin(u1);
        double z2 = cos(u1);
        wx += r * z1;
        wy += r * z2;
    }
    return mark;
}

__device__ unsigned int walk_on_spheres_strided(double x1, double y1,
                                                unsigned int num_shots,
                                                double epsilon,
                                                curandStatePhilox4_32_10_t *state,
                                                unsigned int j_start,
                                                unsigned int j_stride){
    double wx = x1;
    double wy = y1;
    unsigned int mark = 0;
    unsigned int j = j_start;
    while (j < num_shots) {
        if (wx < 0.0) {
            wx = -wx;
        } else if (wx > 1.0){
            wx = 2.0 - wx;
        }
        double r = (HEIGHT - wy) > wy ? wy : (HEIGHT - wy);
        if (r < EPSILON){
            if (wy > HEIGHT/2.0) mark++;
            wx = x1;
            wy = y1;
            j += j_stride;
            continue;
        }
        double u1 = curand_uniform_double(state) * 2.0 * PI;
        double z1 = sin(u1);
        double z2 = cos(u1);
        wx += r * z1;
        wy += r * z2;
    }
    return mark;
}


__global__ void wos_kernel(int n,
                           double x1, double y1,
                           double x2, double y2,
                           double x3, double y3,
                           double x4, double y4,
                           double epsilon,
                           unsigned int num_shots,
                           unsigned int *fm2h_list,
                           unsigned int *fmh_list,
                           unsigned int *fph_list,
                           unsigned int *fp2h_list,
                           unsigned long long seed) {
    int p = blockIdx.x;  // one block per point
    if (p >= n) return;

    int tid = threadIdx.x;

    curandStatePhilox4_32_10_t state;
    curand_init(seed, (unsigned long long)p * blockDim.x + tid, 0, &state);

    unsigned int local_m2 = walk_on_spheres_strided(x1, y1, num_shots, epsilon,
                                                    &state,
                                                    (unsigned int)tid,
                                                    (unsigned int)blockDim.x);
    unsigned int local_m1 = walk_on_spheres_strided(x2, y2, num_shots, epsilon,
                                                    &state,
                                                    (unsigned int)tid,
                                                    (unsigned int)blockDim.x);
    unsigned int local_p1 = walk_on_spheres_strided(x3, y3, num_shots, epsilon,
                                                    &state,
                                                    (unsigned int)tid,
                                                    (unsigned int)blockDim.x);
    unsigned int local_p2 = walk_on_spheres_strided(x4, y4, num_shots, epsilon,
                                                    &state,
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
        fm2h_list[p] = s_m2[0];
        fmh_list[p]  = s_m1[0];
        fph_list[p]  = s_p1[0];
        fp2h_list[p] = s_p2[0];
    }
}

/* ---------------- host code ---------------- */

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

    // Geometry for all points (kept as before)
    double x1 = x_center + (-2) * STEP * cos(ANGLE) ;
    double y1 = y_center + (-2) * STEP * sin(ANGLE) ;
    double x2 = x_center + (-1) * STEP * cos(ANGLE) ;
    double y2 = y_center + (-1) * STEP * sin(ANGLE) ;
    double x3 = x_center + 1 * STEP * cos(ANGLE) ;
    double y3 = y_center + 1 * STEP * sin(ANGLE) ;
    double x4 = x_center + 2 * STEP * cos(ANGLE) ;
    double y4 = y_center + 2 * STEP * sin(ANGLE) ;

    unsigned int *d_fm2h, *d_fmh, *d_fph, *d_fp2h;
    cudaMalloc((void**)&d_fm2h, NUM_POINTS * sizeof(unsigned int));
    cudaMalloc((void**)&d_fmh,  NUM_POINTS * sizeof(unsigned int));
    cudaMalloc((void**)&d_fph,  NUM_POINTS * sizeof(unsigned int));
    cudaMalloc((void**)&d_fp2h, NUM_POINTS * sizeof(unsigned int));

    cudaMemset(d_fm2h, 0, NUM_POINTS * sizeof(unsigned int));
    cudaMemset(d_fmh,  0, NUM_POINTS * sizeof(unsigned int));
    cudaMemset(d_fph,  0, NUM_POINTS * sizeof(unsigned int));
    cudaMemset(d_fp2h, 0, NUM_POINTS * sizeof(unsigned int));

    int n = NUM_POINTS;
    int threads = 256;
    int blocks  = n;
    unsigned long long seed = (unsigned long long)time(NULL);

    size_t smem_size = 4 * threads * sizeof(unsigned int);
    wos_kernel<<<blocks, threads, smem_size>>>(
        n,
        x1, y1,
        x2, y2,
        x3, y3,
        x4, y4,
        EPSILON,
        (unsigned int)NUM_SHOTS,
        d_fm2h, d_fmh, d_fph, d_fp2h,
        seed
    );
    cudaDeviceSynchronize();

    cudaMemcpy(fm2h_list, d_fm2h, NUM_POINTS * sizeof(unsigned int), cudaMemcpyDeviceToHost);
    cudaMemcpy(fmh_list,  d_fmh,  NUM_POINTS * sizeof(unsigned int), cudaMemcpyDeviceToHost);
    cudaMemcpy(fph_list,  d_fph,  NUM_POINTS * sizeof(unsigned int), cudaMemcpyDeviceToHost);
    cudaMemcpy(fp2h_list, d_fp2h, NUM_POINTS * sizeof(unsigned int), cudaMemcpyDeviceToHost);

    cudaFree(d_fm2h);
    cudaFree(d_fmh);
    cudaFree(d_fph);
    cudaFree(d_fp2h);

    print_fm2h_results(fm2h_list, NUM_POINTS);
    print_fmh_results(fmh_list, NUM_POINTS);
    print_fph_results(fph_list, NUM_POINTS);
    print_fp2h_results(fp2h_list, NUM_POINTS);
    time( &rawtime );
    timeinfo = localtime( &rawtime );
    return 0;
}

/* ---------------- original CPU helpers (kept) ---------------- */

inline unsigned int walk_on_spheres(double x1, double y1, unsigned int num_shots, double epsilon){
        double wx = x1;
        double wy = y1;
        unsigned int mark = 0;
        unsigned int j = 0;
        while (j < num_shots) {
                if (wx < 0) {
                        wx = -wx;
                }else if (wx > 1){
                        wx = 2 - wx;
                }
                double r = (HEIGHT-wy) > wy ? wy : (HEIGHT-wy);
                if(r<EPSILON){
                        if(wy > HEIGHT/2) mark++;
                        wx = x1;
                        wy = y1;
                        j++;
                }
                double u1 = ((double)(rand()))*2*PI/(RAND_MAX);
                double z1 = sin(u1);
                double z2 = cos(u1);
                wx += r * z1;
                wy += r * z2;
        }
        return mark;
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
        fprintf(fd, "%u\n", *(list+i) );
    }
    fclose(fd);
}
inline void print_fmh_results(unsigned int* list, int num_of_pts) {
    FILE* fd = fopen("cfmh.txt", "w");
    for (int i = 0; i < num_of_pts; i++) {    
        fprintf(fd, "%u\n", *(list+i) );
    }
    fclose(fd);
}
inline void print_fph_results(unsigned int* list, int num_of_pts) {
    FILE* fd = fopen("cfph.txt", "w");
    for (int i = 0; i < num_of_pts; i++) {    
        fprintf(fd, "%u\n", *(list+i) );
    }
    fclose(fd);
}
inline void print_fp2h_results(unsigned int* list, int num_of_pts) {
    FILE* fd = fopen("cfp2h.txt", "w");
    for (int i = 0; i < num_of_pts; i++) {    
        fprintf(fd, "%u\n", *(list+i) );
    }
    fclose(fd);
}
