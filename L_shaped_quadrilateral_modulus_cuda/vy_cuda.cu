// vy_cuda.cu
// CUDA version of vy.c (vertical finite-difference samples for vy)
// Build: nvcc -O3 -arch=sm_70 vy_cuda.cu -o vy_cuda -lcurand
// Run:   ./vy_cuda

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <ctime>
#include <cstring>

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

#define PI 3.14159265358979323846
#define x_center 1.9
#define y_center 0.9
#define STEP 0.1           // step size h
#define ANGLE (PI / 2.0) 
#define NUM_POINTS 1000
#define NUM_SHOTS 1.0E6   
#define EPSILON 1e-6
#define length(x, y) sqrt((x)*(x) + (y)*(y))


inline void print_fm2h_results(unsigned int* list, int num_of_pts) {
    const char name[] = "cfm2h.txt";
    FILE* fd = fopen(name, "w");
    if (!fd) { perror("fopen fm2h"); return; }
    for (int i = 0; i < num_of_pts; i++) {
        fprintf(fd, "%u\n", list[i]);
    }
    fclose(fd);
}

inline void print_fmh_results(unsigned int* list, int num_of_pts) {
    const char name[] = "cfmh.txt";
    FILE* fd = fopen(name, "w");
    if (!fd) { perror("fopen fmh"); return; }
    for (int i = 0; i < num_of_pts; i++) {
        fprintf(fd, "%u\n", list[i]);
    }
    fclose(fd);
}

inline void print_fph_results(unsigned int* list, int num_of_pts) {
    const char name[] = "cfph.txt";
    FILE* fd = fopen(name, "w");
    if (!fd) { perror("fopen fph"); return; }
    for (int i = 0; i < num_of_pts; i++) {
        fprintf(fd, "%u\n", list[i]);
    }
    fclose(fd);
}

inline void print_fp2h_results(unsigned int* list, int num_of_pts) {
    const char name[] = "cfp2h.txt";
    FILE* fd = fopen(name, "w");
    if (!fd) { perror("fopen fp2h"); return; }
    for (int i = 0; i < num_of_pts; i++) {
        fprintf(fd, "%u\n", list[i]);
    }
    fclose(fd);
}

__device__ unsigned int walk_on_spheres_dev(double x_start,
                                            double y_start,
                                            unsigned int num_shots,
                                            double epsilon,
                                            curandStatePhilox4_32_10_t* rng)
{
    unsigned int mark = 0;

    for (unsigned int j = 0; j < num_shots; ++j) {
        double x = x_start;
        double y = y_start;

        while (true) {
       
            if (x < 0.0) {
                x = -x;
            } else if (x > 6.0) {
                x = x - 6.0;
            } else if (2.0 < x && x < 4.0 && y > 1.0) {
                y = 2.0 - y;
            }

            double r = 0.0;
            if (y <= 1.0) {
                double dist1 = y;
                double dist2 = length(x - 2.0, y - 1.0);
                double dist3 = length(x - 4.0, y - 1.0);
                r = (dist1 < dist2)
                        ? ((dist1 < dist3) ? dist1 : dist3)
                        : ((dist2 < dist3) ? dist2 : dist3);
            } else {
                if (x < 2.0) {
                    r = ((2.0 - y) < (2.0 - x)) ? (2.0 - y) : (2.0 - x);
                } else if (x > 4.0) {
                    r = ((2.0 - y) < (x - 4.0)) ? (2.0 - y) : (x - 4.0);
                } else {
                    r = ((x - 2.0) < (4.0 - x)) ? (x - 2.0) : (4.0 - x);
                }
            }

            if (r < epsilon) {
                if (y > 1.0) {
                    ++mark;
                }
                break;
            }

            double u = curand_uniform_double(rng);  // (0,1]
            double theta = 2.0 * PI * u;
            double z1 = sin(theta);
            double z2 = cos(theta);
            x += r * z1;
            y += r * z2;
        } // while
    }     // for shots

    return mark;
}

__device__ unsigned int walk_on_spheres_dev_strided(double x_start,
                                                    double y_start,
                                                    unsigned int num_shots,
                                                    double epsilon,
                                                    curandStatePhilox4_32_10_t* rng,
                                                    unsigned int j_start,
                                                    unsigned int j_stride)
{
    unsigned int mark = 0;

    for (unsigned int j = j_start; j < num_shots; j += j_stride) {
        double x = x_start;
        double y = y_start;

        while (true) {
            if (x < 0.0) {
                x = -x;
            } else if (x > 6.0) {
                x = x - 6.0;
            } else if (2.0 < x && x < 4.0 && y > 1.0) {
                y = 2.0 - y;
            }

            double r = 0.0;
            if (y <= 1.0) {
                double dist1 = y;
                double dist2 = length(x - 2.0, y - 1.0);
                double dist3 = length(x - 4.0, y - 1.0);
                r = (dist1 < dist2)
                        ? ((dist1 < dist3) ? dist1 : dist3)
                        : ((dist2 < dist3) ? dist2 : dist3);
            } else {
                if (x < 2.0) {
                    r = ((2.0 - y) < (2.0 - x)) ? (2.0 - y) : (2.0 - x);
                } else if (x > 4.0) {
                    r = ((2.0 - y) < (x - 4.0)) ? (2.0 - y) : (x - 4.0);
                } else {
                    r = ((x - 2.0) < (4.0 - x)) ? (x - 2.0) : (4.0 - x);
                }
            }

            if (r < epsilon) {
                if (y > 1.0) {
                    ++mark;
                }
                break;
            }

            double u = curand_uniform_double(rng);  // (0,1]
            double theta = 2.0 * PI * u;
            double z1 = sin(theta);
            double z2 = cos(theta);
            x += r * z1;
            y += r * z2;
        }
    }

    return mark;
}

// --------------------- CUDA kernel: block-per-point with strided shots and shared-memory reduction ---------------------

__global__ void vy_kernel(int n,
                          double x1, double y1,
                          double x2, double y2,
                          double x3, double y3,
                          double x4, double y4,
                          unsigned int num_shots,
                          double epsilon,
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

    unsigned int local_m2 = walk_on_spheres_dev_strided(x1, y1, num_shots, epsilon, &rng,
                                                        (unsigned int)tid, (unsigned int)blockDim.x);
    unsigned int local_m1 = walk_on_spheres_dev_strided(x2, y2, num_shots, epsilon, &rng,
                                                        (unsigned int)tid, (unsigned int)blockDim.x);
    unsigned int local_p1 = walk_on_spheres_dev_strided(x3, y3, num_shots, epsilon, &rng,
                                                        (unsigned int)tid, (unsigned int)blockDim.x);
    unsigned int local_p2 = walk_on_spheres_dev_strided(x4, y4, num_shots, epsilon, &rng,
                                                        (unsigned int)tid, (unsigned int)blockDim.x);


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

// --------------------- main ---------------------

int main() {
    time_t rawtime;
    struct tm* timeinfo;
    time(&rawtime);
    timeinfo = localtime(&rawtime);
    printf("Current local time and date: %s", asctime(timeinfo));
    printf("Every point has %lf shots\n", NUM_SHOTS);

    unsigned int fm2h_list[NUM_POINTS] = {0};
    unsigned int fmh_list[NUM_POINTS]  = {0};
    unsigned int fph_list[NUM_POINTS]  = {0};
    unsigned int fp2h_list[NUM_POINTS] = {0};

    int N = NUM_POINTS;

    double x1 = x_center + (-2.0) * STEP * cos(ANGLE);
    double y1 = y_center + (-2.0) * STEP * sin(ANGLE);
    double x2 = x_center + (-1.0) * STEP * cos(ANGLE);
    double y2 = y_center + (-1.0) * STEP * sin(ANGLE);
    double x3 = x_center + ( 1.0) * STEP * cos(ANGLE);
    double y3 = y_center + ( 1.0) * STEP * sin(ANGLE);
    double x4 = x_center + ( 2.0) * STEP * cos(ANGLE);
    double y4 = y_center + ( 2.0) * STEP * sin(ANGLE);

    unsigned int *d_fm2h = nullptr, *d_fmh = nullptr;
    unsigned int *d_fph  = nullptr, *d_fp2h = nullptr;

    cudaMalloc((void**)&d_fm2h, N * sizeof(unsigned int));
    cudaMalloc((void**)&d_fmh,  N * sizeof(unsigned int));
    cudaMalloc((void**)&d_fph,  N * sizeof(unsigned int));
    cudaMalloc((void**)&d_fp2h, N * sizeof(unsigned int));

    const int TPB = 256;
    const int blocks = N;
    size_t shmem_size = 4 * TPB * sizeof(unsigned int);

    unsigned long long seed = (unsigned long long)time(NULL);

    vy_kernel<<<blocks, TPB, shmem_size>>>(N,
                                           x1, y1,
                                           x2, y2,
                                           x3, y3,
                                           x4, y4,
                                           (unsigned int)NUM_SHOTS,
                                           EPSILON,
                                           d_fm2h, d_fmh, d_fph, d_fp2h,
                                           seed);
    cudaDeviceSynchronize();

    cudaMemcpy(fm2h_list, d_fm2h, N * sizeof(unsigned int), cudaMemcpyDeviceToHost);
    cudaMemcpy(fmh_list,  d_fmh,  N * sizeof(unsigned int), cudaMemcpyDeviceToHost);
    cudaMemcpy(fph_list,  d_fph,  N * sizeof(unsigned int), cudaMemcpyDeviceToHost);
    cudaMemcpy(fp2h_list, d_fp2h, N * sizeof(unsigned int), cudaMemcpyDeviceToHost);

    print_fm2h_results(fm2h_list, N);
    print_fmh_results (fmh_list,  N);
    print_fph_results (fph_list,  N);
    print_fp2h_results(fp2h_list, N);

    cudaFree(d_fm2h);
    cudaFree(d_fmh);
    cudaFree(d_fph);
    cudaFree(d_fp2h);

    time(&rawtime);
    timeinfo = localtime(&rawtime);
    printf("Finish time and date: %s", asctime(timeinfo));

    return 0;
}
