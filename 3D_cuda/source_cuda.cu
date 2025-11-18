// source_cuda.cu
// Build: nvcc -O3 -arch=sm_70 source_cuda.cu -o source_cuda -lcurand
// Usage: ./source_cuda

#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <math.h>
#include <time.h>
#include <string.h>

#include <cuda_runtime.h>
#include <curand_kernel.h>

typedef enum {
    one, two, three
} domainLabel;

typedef struct {
    double x;
    double y;
    double z;
    double result;
    domainLabel label;
} point_t, *Point;

/* Volume discretization parameters */
#define X_SLIDES 12
#define Y_SLIDES 10
#define Z_SLIDES 20
#define NUM_POINTS (X_SLIDES * Y_SLIDES * Z_SLIDES)
#define NUM_SHOTS 100000.0 /* Must be a float number */
#define FAILURE_RADIUS 1e-5 /* When radius less than this, it fails, we try again */
#define FAILURE_BOUND 500 /* (unused, kept for consistency) */
#define PI 3.14159
#define PRECISE 1e-6

#define length(x, y) sqrt((x)*(x) + (y)*(y))

inline Point create_point(double, double, double);
inline Point* get_points(int*);
inline void destroy_points(Point*, int);
inline void print_points(Point*, int);

// ---------------- CUDA error check ----------------
#define CHECK_CUDA(call)                                                      \
    do {                                                                      \
        cudaError_t err__ = (call);                                           \
        if (err__ != cudaSuccess) {                                           \
            fprintf(stderr, "CUDA error %s at %s:%d\n",                       \
                    cudaGetErrorString(err__), __FILE__, __LINE__);           \
            exit(EXIT_FAILURE);                                               \
        }                                                                     \
    } while (0)

// ---------------- device-side helpers (3D WoS) ----------------
__device__ inline double length3(double x, double y, double z) {
    return sqrt(x * x + y * y + z * z);
}

__device__ double find_min_radius_dev(double x, double y, double z, domainLabel *label) {
    double l1, l2;
    if (z >= 1.0 && x <= 1.0) {
        l1 = 2.0 - z;
        l2 = sqrt((1.0 - x) * (1.0 - x) + (1.0 - z) * (1.0 - z));
        *label = one;
    } else if (x > 1.0) {
        l1 = 3.0 - x;
        l2 = sqrt((1.0 - x) * (1.0 - x) + (1.0 - z) * (1.0 - z));
        *label = three;
    } else {
        l1 = sqrt((1.0 - x) * (1.0 - x) + (1.0 - z) * (1.0 - z));
        l2 = l1;
        *label = two;
    }
    return (l1 >= l2) ? l2 : l1;
}

__device__ void move_to_next_dev(double &x, double &y, double &z,
                                 curandStatePhilox4_32_10_t *rng,
                                 double r) {
    double u1 = curand_uniform_double(rng);
    double u2 = curand_uniform_double(rng);
    double u3 = curand_uniform_double(rng);
    double u4 = curand_uniform_double(rng);

    double z1 = sqrt(-2.0 * log(u1)) * cos(2.0 * PI * u2);
    double z2 = sqrt(-2.0 * log(u1)) * sin(2.0 * PI * u2);
    double z3 = sqrt(-2.0 * log(u3)) * cos(2.0 * PI * u4);
    double norm = sqrt(z1 * z1 + z2 * z2 + z3 * z3);

    double nx = z1 / norm;
    double ny = z2 / norm;
    double nz = z3 / norm;

    x += r * nx;
    y += r * ny;
    z += r * nz;
}

__device__ void bounce_inside_dev(double &x, double &y, double &z, domainLabel *label) {
    // Geometric reflection to keep (x, y, z) inside the L-shaped domain
    while (! (0.0 <= x && x <= 3.0 &&
              0.0 <= y && y <= 1.0 &&
              0.0 <= z && z <= 2.0 &&
              ! (1.0 < x && x <= 3.0 && 0.0 <= y && y <= 1.0 && 1.0 < z && z <= 2.0))) {

        if (x < 0.0) x = -x;
        if (y < 0.0) y = -y;
        if (y > 1.0) y = 2.0 - y;
        if (z < 0.0) z = -z;

        if (x >= 1.0 && x <= 3.0 && z >= 1.0 && z <= 2.0) {
            if (*label == one)
                x = 2.0 - x;
            else
                z = 2.0 - z;
        }
    }

    if (x >= 1.0) *label = three;
    else if (z >= 1.0) *label = one;
    else *label = two;
}

__device__ bool isOnPlanes_dev(double x, double z, int *mark) {
    if (x >= 3.0 - PRECISE) {
        (*mark)++;
        return true;
    } else if (z >= 2.0 - PRECISE) {
        return true;
    }
    return false;
}

__device__ unsigned int wos3d_walk_strided(double x0,
                                           double y0,
                                           double z0,
                                           domainLabel label0,
                                           unsigned int num_shots,
                                           curandStatePhilox4_32_10_t* rng,
                                           unsigned int j_start,
                                           unsigned int j_stride)
{
    unsigned int mark = 0;

    for (unsigned int j = j_start; j < num_shots; j += j_stride) {
        double x = x0;
        double y = y0;
        double z = z0;
        domainLabel label = label0;

        while (true) {
            // FAILURE_RADIUS check consistent with the original CPU implementation
            if (length(1.0 - x, 1.0 - z) < FAILURE_RADIUS) {
                x = x0;
                y = y0;
                z = z0;
                label = label0;
            }

            double r = find_min_radius_dev(x, y, z, &label);
            move_to_next_dev(x, y, z, rng, r);
            bounce_inside_dev(x, y, z, &label);

            int mark_int = (int)mark;
            if (isOnPlanes_dev(x, z, &mark_int)) {
                mark = (unsigned int)mark_int;
                break;
            } else {
                mark = (unsigned int)mark_int;
            }
        }
    }

    return mark;
}

// ---------------- CUDA kernel: block-per-point, strided shots, shared-memory reduction ----------------
__global__ void wos3d_kernel(const double *xs,
                             const double *ys,
                             const double *zs,
                             const int    *labels0,
                             double       *results,
                             int n,
                             unsigned long long seed) {
    int p = blockIdx.x;  // Point index [0, n)
    if (p >= n) return;

    int tid = threadIdx.x;

    double x0 = xs[p];
    double y0 = ys[p];
    double z0 = zs[p];
    domainLabel label0 = (domainLabel)labels0[p];

    curandStatePhilox4_32_10_t rng;
    curand_init(seed, (unsigned long long)p * blockDim.x + tid, 0ULL, &rng);

    const unsigned int shots = (unsigned int)NUM_SHOTS;

    // Each thread handles a strided subset of shots
    unsigned int local_mark = wos3d_walk_strided(x0, y0, z0, label0,
                                                 shots,
                                                 &rng,
                                                 (unsigned int)tid,
                                                 (unsigned int)blockDim.x);

    // Shared-memory reduction of per-thread hit counts
    extern __shared__ unsigned int sdata[];
    sdata[tid] = local_mark;
    __syncthreads();

    for (int offset = blockDim.x >> 1; offset > 0; offset >>= 1) {
        if (tid < offset) {
            sdata[tid] += sdata[tid + offset];
        }
        __syncthreads();
    }

    if (tid == 0) {
        results[p] = (double)sdata[0] / NUM_SHOTS;
    }
}

#ifndef TEST
int main() {
    time_t rawtime;
    struct tm * timeinfo;
    time(&rawtime);
    timeinfo = localtime(&rawtime);
    printf("Current local time and date: %s", asctime(timeinfo));
    printf("Every point has %lf shots\n", NUM_SHOTS);

    int num_of_pts;
    Point* list = get_points(&num_of_pts);
    srand((unsigned)time(NULL));

    int N = num_of_pts;

    // host arrays
    double *hx = (double*)malloc(N * sizeof(double));
    double *hy = (double*)malloc(N * sizeof(double));
    double *hz = (double*)malloc(N * sizeof(double));
    int    *hlabel = (int*)malloc(N * sizeof(int));
    double *hres   = (double*)malloc(N * sizeof(double));

    for (int i = 0; i < N; ++i) {
        Point p = list[i];
        hx[i] = p->x;
        hy[i] = p->y;
        hz[i] = p->z;
        hlabel[i] = (int)p->label;
        hres[i] = 0.0;
    }

    // device arrays
    double *dx = NULL, *dy = NULL, *dz = NULL, *dres = NULL;
    int    *dlabel = NULL;

    CHECK_CUDA(cudaMalloc((void**)&dx,     N * sizeof(double)));
    CHECK_CUDA(cudaMalloc((void**)&dy,     N * sizeof(double)));
    CHECK_CUDA(cudaMalloc((void**)&dz,     N * sizeof(double)));
    CHECK_CUDA(cudaMalloc((void**)&dres,   N * sizeof(double)));
    CHECK_CUDA(cudaMalloc((void**)&dlabel, N * sizeof(int)));

    CHECK_CUDA(cudaMemcpy(dx, hx, N * sizeof(double), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dy, hy, N * sizeof(double), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dz, hz, N * sizeof(double), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dlabel, hlabel, N * sizeof(int), cudaMemcpyHostToDevice));

    const int TPB    = 256;
    const int blocks = N;  // One block per point
    unsigned long long seed = (unsigned long long)time(NULL);
    size_t shmem_size = TPB * sizeof(unsigned int);

    wos3d_kernel<<<blocks, TPB, shmem_size>>>(dx, dy, dz, dlabel, dres, N, seed);
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaMemcpy(hres, dres, N * sizeof(double), cudaMemcpyDeviceToHost));

    for (int i = 0; i < N; ++i) {
        list[i]->result = hres[i];
    }

    print_points(list, N);
    destroy_points(list, num_of_pts);

    CHECK_CUDA(cudaFree(dx));
    CHECK_CUDA(cudaFree(dy));
    CHECK_CUDA(cudaFree(dz));
    CHECK_CUDA(cudaFree(dres));
    CHECK_CUDA(cudaFree(dlabel));

    free(hx);
    free(hy);
    free(hz);
    free(hlabel);
    free(hres);

    time(&rawtime);
    timeinfo = localtime(&rawtime);
    printf("Finish time and date: %s\n", asctime(timeinfo));
    return 0;
}
#endif

// ---------------- Host-side helpers ----------------

inline Point create_point(double x, double y, double z) {
    Point newp = (Point)malloc(sizeof(*newp));
    newp->x = x;
    newp->y = y;
    newp->z = z;
    newp->result = 0.0;
    if (newp->x >= 1.0) newp->label = three;
    else if (newp->z >= 1.0) newp->label = one;
    else newp->label = two;
    return newp;
}

inline Point* get_points(int* num_of_pts) {
    int counter = 0;
    double x_d = 3.0 / X_SLIDES;
    double y_d = 1.0 / Y_SLIDES;
    double z_d = 2.0 / Z_SLIDES;
    Point* list = (Point*)malloc(sizeof(*list) * NUM_POINTS);
    for (int i = 0; i < NUM_POINTS; i++) list[i] = NULL;

    for (int i = 0; i <= X_SLIDES; i++) {
        double x = i * x_d;
        for (int j = 0; j <= Y_SLIDES; j++) {
            double y = j * y_d;
            for (int k = 0; k <= Z_SLIDES; k++) {
                double z = k * z_d;
                if (x > 1.0 && z > 1.0) continue;
                else if (length(1.0 - x, 1.0 - z) < 1e-3) continue;
                list[counter++] = create_point(x, y, z);
            }
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
    if (!fd) return;
    for (int i = 0; i < num_of_pts; i++) {
        Point pt = list[i];
        fprintf(fd, "%1.6lf, %1.6lf, %1.6lf, %1.6lf\n",
                pt->x, pt->y, pt->z, pt->result);
    }
    fclose(fd);
}
