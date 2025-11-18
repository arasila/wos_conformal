// source.cu
// CUDA Walk-on-Spheres in circular domain with two holes
// Build: nvcc -O3 -arch=sm_89 --use_fast_math source.cu -o source -lcurand
// Run:   ./source

#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <math.h>
#include <time.h>
#include <string.h>

#include <cuda_runtime.h>
#include <curand_kernel.h>

/* Point structure */
typedef struct {
    double x;
    double y;
    double result;
    unsigned int num_shots;
} point_t, *Point;

/* Grid discretization */
#define Y_SLIDES 100
#define X_SLIDES 100
#define NUM_POINTS ((X_SLIDES + 1) * (Y_SLIDES + 1))

#define NUM_SHOTS 100000.0 /* Must be a float number */
#define PI 3.14159265358979323846
#define EPSILON 1e-6
#define length2D(x, y) sqrt((x)*(x) + (y)*(y))

/* Geometric parameters */
#define center2x -1.164524664599176
#define center2y -0.4823619097949585
#define r2 0.7673269879789604
#define center4x 1.0
#define center4y 0.5773502691896258
#define r4 0.5773502691896258
#define Tarc2Centerx 0.8617632503267921
#define Tarc2Centery 0.509672053455795
#define Tarc2Radius 0.04900511899428376
#define Tarc4Centerx -0.9317405419866858
#define Tarc4Centery -0.3683950080353163
#define Tarc4Radius 0.06209121940325892

/* ---- CUDA error check ---- */
#define CHECK_CUDA(call)                                                      \
    do {                                                                      \
        cudaError_t err__ = (call);                                           \
        if (err__ != cudaSuccess) {                                           \
            fprintf(stderr, "CUDA error %s at %s:%d\n",                       \
                    cudaGetErrorString(err__), __FILE__, __LINE__);           \
            exit(EXIT_FAILURE);                                               \
        }                                                                     \
    } while (0)

/* Host-side declarations */
inline Point create_point(double, double);
inline Point* get_points(int*);
inline void destroy_points(Point*, int);
inline void print_points(Point*, int);

/* Domain membership test */
inline bool inside_domain(double x, double y) {
    if (x*x + y*y > 1.0) return false;
    if (sqrt((x - center2x)*(x - center2x) + (y - center2y)*(y - center2y)) < r2) return false;
    if (sqrt((x - center4x)*(x - center4x) + (y - center4y)*(y - center4y)) < r4) return false;
    return true;
}

/* Device-side Walk-on-Spheres */
__device__ double walk_on_spheres_dev(double x1,
                                      double y1,
                                      unsigned int NN,
                                      double eps,
                                      curandStatePhilox4_32_10_t* rng)
{
    double wx = x1;
    double wy = y1;
    unsigned int cnt = 0;
    unsigned int j = 0;
    double distance, theta, denom;

    while (j < NN) {
        denom = (wx * wx + wy * wy);
        if (denom > 1.0) {
           
            wx = wx / denom;
            wy = wy / denom;
        }

        double dist1 = length2D(wx - center2x, wy - center2y) - r2;
        double dist2 = length2D(wx - center4x, wy - center4y) - r4;
        distance = (dist1 > dist2) ? dist2 : dist1;

        if (distance < eps) {
            cnt += (wx < 0.0) ? 0u : 1u;
            wx = x1;
            wy = y1;
            ++j;
        } else {
            double u = curand_uniform_double(rng);  // (0,1]
            theta = 2.0 * PI * u;
            wx += distance * cos(theta);
            wy += distance * sin(theta);
        }
    }

    return ((double)cnt) / (double)NN;
}

/* CUDA kernel: one thread per point */
__global__ void wos_kernel(const double* __restrict__ xs,
                           const double* __restrict__ ys,
                           const unsigned int* __restrict__ shots,
                           double* __restrict__ results,
                           int N,
                           double eps,
                           unsigned long long seed)
{
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid >= N) return;

    double x = xs[tid];
    double y = ys[tid];
    unsigned int num_shots = shots[tid];

    curandStatePhilox4_32_10_t rng;
    curand_init(seed, (unsigned long long)tid, 0ULL, &rng);

    double val = walk_on_spheres_dev(x, y, num_shots, eps, &rng);
    results[tid] = val;
}

/* Main function using CUDA */
int main() {

    time_t rawtime;
    struct tm * timeinfo;
    time(&rawtime);
    timeinfo = localtime(&rawtime);
    printf("Current local time and date: %s", asctime(timeinfo));
    printf("Every point has %lf shots\n", NUM_SHOTS);

    int num_of_pts;
    Point* list = get_points(&num_of_pts);
    srand((unsigned int)time(NULL));  

    int N = num_of_pts;

    double* hx    = (double*)malloc(N * sizeof(double));
    double* hy    = (double*)malloc(N * sizeof(double));
    double* hres  = (double*)malloc(N * sizeof(double));
    unsigned int* hshots = (unsigned int*)malloc(N * sizeof(unsigned int));
    if (!hx || !hy || !hres || !hshots) {
        fprintf(stderr, "Host malloc failed\n");
        destroy_points(list, num_of_pts);
        return -1;
    }

    for (int i = 0; i < N; ++i) {
        Point pt = list[i];
        hx[i]    = pt->x;
        hy[i]    = pt->y;
        hshots[i]= pt->num_shots;
    }

    double *dx = NULL, *dy = NULL, *dres = NULL;
    unsigned int* dshots = NULL;

    CHECK_CUDA(cudaMalloc((void**)&dx,    N * sizeof(double)));
    CHECK_CUDA(cudaMalloc((void**)&dy,    N * sizeof(double)));
    CHECK_CUDA(cudaMalloc((void**)&dres,  N * sizeof(double)));
    CHECK_CUDA(cudaMalloc((void**)&dshots,N * sizeof(unsigned int)));

    CHECK_CUDA(cudaMemcpy(dx, hx,     N * sizeof(double),       cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dy, hy,     N * sizeof(double),       cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dshots,hshots,N * sizeof(unsigned int), cudaMemcpyHostToDevice));

    const int TPB    = 256;
    const int blocks = (N + TPB - 1) / TPB;
    unsigned long long seed = (unsigned long long)time(NULL);

    wos_kernel<<<blocks, TPB>>>(dx, dy, dshots, dres, N, EPSILON, seed);
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaMemcpy(hres, dres, N * sizeof(double), cudaMemcpyDeviceToHost));

    for (int i = 0; i < N; ++i) {
        list[i]->result = hres[i];
    }

    print_points(list, N);
    destroy_points(list, num_of_pts);

    free(hx);
    free(hy);
    free(hres);
    free(hshots);

    CHECK_CUDA(cudaFree(dx));
    CHECK_CUDA(cudaFree(dy));
    CHECK_CUDA(cudaFree(dres));
    CHECK_CUDA(cudaFree(dshots));

    time(&rawtime);
    timeinfo = localtime(&rawtime);
    printf("Finish time and date: %s", asctime(timeinfo));

    return 0;
}


inline Point create_point(double x, double y) {
    Point newp = (Point)malloc(sizeof(*newp));
    if (!newp) {
        perror("Failed to allocate point");
        exit(EXIT_FAILURE);
    }
    newp->x = x;
    newp->y = y;
    newp->result = 0.0;
    newp->num_shots = (unsigned int)NUM_SHOTS;
    return newp;
}

inline Point* get_points(int* num_of_pts) {
    int counter = 0;
    double x_d = 2.0 / X_SLIDES;
    double y_d = 2.0 / Y_SLIDES;
    double x, y;

    int list_Size = NUM_POINTS;
    Point* list = (Point*)malloc(sizeof(*list) * list_Size);
    if (!list) {
        perror("Failed to allocate memory");
        exit(EXIT_FAILURE);
    }

    for (int i = 0; i < list_Size; i++) list[i] = NULL;

    // Smoothen the boundary (two circular arcs sampling)
    for (int i = 0; i <= 500; i++) {
        if (counter >= list_Size) {
            list_Size *= 2;
            list = (Point*)realloc(list, sizeof(*list) * list_Size);
            if (!list) {
                perror("Failed to reallocate memory");
                exit(EXIT_FAILURE);
            }
        }
        double dtheta = ((11.0 / 12.0) - (1.0 / 3.0)) * PI / 500.0;
        double t = PI / 3.0 + i * dtheta;
        x = cos(t);
        y = sin(t);
        list[counter] = create_point(x, y);
        counter++;
    }

    for (int i = 0; i <= 500; i++) {
        if (counter >= list_Size) {
            list_Size *= 2;
            list = (Point*)realloc(list, sizeof(*list) * list_Size);
            if (!list) {
                perror("Failed to reallocate memory");
                exit(EXIT_FAILURE);
            }
        }
        double dtheta = (2.0 / 3.0) * PI / 500.0;
        double t = (-2.0 / 3.0) * PI + i * dtheta;
        x = cos(t);
        y = sin(t);
        list[counter] = create_point(x, y);
        counter++;
    }

    for (int i = 0; i <= X_SLIDES; i++) {
        x = -1.0 + i * x_d;
        for (int j = 0; j <= Y_SLIDES; j++) {
            if (counter >= list_Size) {
                list_Size *= 2;
                list = (Point*)realloc(list, sizeof(*list) * list_Size);
                if (!list) {
                    perror("Failed to reallocate memory");
                    exit(EXIT_FAILURE);
                }
            }
            y = -1.0 + j * y_d;
            if (!inside_domain(x, y)) continue;
            list[counter] = create_point(x, y);
            counter++;
        }
    }

    *num_of_pts = counter;
    return list;
}

inline void destroy_points(Point* list, int num_of_pts) {
    for (int i = 0; i < num_of_pts; i++)
        free(list[i]);
    free(list);
}

inline void print_points(Point* list, int num_of_pts) {
    const char name[] = "points.txt";
    FILE* fd = fopen(name, "w");
    if (!fd) {
        perror("fopen points");
        return;
    }
    for (int i = 0; i < num_of_pts; i++) {
        Point pt = list[i];
        fprintf(fd, "%lf, %lf, %lf\n", pt->x, pt->y, pt->result);
    }
    fclose(fd);
}
