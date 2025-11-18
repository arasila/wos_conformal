// wos_cuda.cu
// CUDA port of source.c (Walk-on-Spheres with mixed Neumann/Dirichlet BCs)
// nvcc -O3 -arch=sm_89 --use_fast_math source.cu -o source -lcurand
// Run:   ./wos_cuda

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <ctime>
#include <cstring>
#include <vector>
#include <curand_kernel.h>

typedef struct {
    double x;
    double y;
    double result;
    unsigned int num_shots;
} point_t, *Point;

/* --- constants / macros kept consistent with source.c --- */
#define X_SLIDES 100
#define Y_SLIDES 100
#define NUM_POINTS (X_SLIDES * Y_SLIDES)
#define NUM_SHOTS_PER_PT 5000000u  // use uint here (source.c uses a float macro but传入的是unsigned int) :contentReference[oaicite:3]{index=3}
#define PI 3.14159265358979323846
#define EPSILON 1e-6
#define length2D(x, y) sqrt((x)*(x) + (y)*(y))

static inline Point create_point(double x, double y) {
    Point p = (Point)malloc(sizeof(*p));
    p->x = x; p->y = y; p->result = 0.0; p->num_shots = NUM_SHOTS_PER_PT;
    return p;
}

static inline Point* get_points(int* num_of_pts) {
    int counter = 0;
    double x_d = 3.0 / X_SLIDES;
    double y_d = 2.0 / Y_SLIDES;

    int list_Size = NUM_POINTS;
    Point* list = (Point*)malloc(sizeof(*list) * list_Size);
    if (!list) { perror("malloc failed"); exit(1); }
    for (int i = 0; i < list_Size; ++i) list[i] = NULL;

    // base grid
    for (int i = 0; i <= X_SLIDES; ++i) {
        double x = i * x_d;
        for (int j = 0; j <= Y_SLIDES; ++j) {
            double y = j * y_d;
            if (x > 2 && y > 1) continue;
            if (counter >= list_Size) {
                list_Size *= 2;
                list = (Point*)realloc(list, sizeof(*list) * list_Size);
                if (!list) { perror("realloc failed"); exit(1); }
            }
            list[counter] = create_point(x, y);
            ++counter;
        }
    }

    for (int i = 0; i <= 20; ++i) {
        double x = 1.9 + (i + 1) * 0.01;
        for (int j = 0; j <= 20; ++j) {
            double y = 0.95 + (j + 1) * 0.0101;
            if (x > 2 && y > 1) continue;
            if (counter >= list_Size) {
                list_Size *= 2;
                list = (Point*)realloc(list, sizeof(*list) * list_Size);
                if (!list) { perror("realloc failed"); exit(1); }
            }
            list[counter] = create_point(x, y);
            ++counter;
        }
    }

    *num_of_pts = counter;
    return list;
}

static inline void destroy_points(Point* list, int num_of_pts) {
    for (int i = 0; i < num_of_pts; ++i) free(list[i]);
    free(list);
}

static inline void print_points(Point* list, int num_of_pts) {
    const char name[] = "points.txt";
    FILE* fd = fopen(name, "w");
    if (!fd) { perror("fopen"); return; }
    for (int i = 0; i < num_of_pts; ++i) {
        fprintf(fd, "%lf, %lf, %lf\n", list[i]->x, list[i]->y, list[i]->result);
    }
    fclose(fd);
}

/* --------------------- device-side math / BC helpers --------------------- */
__device__ __forceinline__ void reflect_inside(double &x, double &y) {
    if (x < 0.0)        x = -x;       // Neumann at x=0
    else if (x > 6.0)   x = x - 6.0;  // Neumann at x=6 (mirror/periodic-like treatment in original)
    else if (x > 2.0 && x < 4.0 && y > 1.0)
        y = 2.0 - y;                  // Neumann at y=1 for 2<=x<=4
}

__device__ __forceinline__ double find_min_radius_dev(double x, double y) {
    if (y <= 1.0) {
        double dist1 = y;
        double dist2 = length2D(x - 2.0, y - 1.0);
        double dist3 = length2D(x - 4.0, y - 1.0);
        return (dist1 < dist2) ? ((dist1 < dist3) ? dist1 : dist3)
                               : ((dist2 < dist3) ? dist2 : dist3);
    } else {
        if (x < 2.0)       return fmin(2.0 - y, 2.0 - x);
        else if (x > 4.0)  return fmin(2.0 - y, x - 4.0);
        else               return fmin(x - 2.0, 4.0 - x);
    }
}

__device__ __forceinline__ bool hit_dirichlet_and_is_top(double x, double y, double eps) {

    return (y > 1.0); 
}

/* --------------------- strided WoS helper for block-per-point kernel --------------------- */
__device__ unsigned int wos_walk_strided(double x0,
                                         double y0,
                                         unsigned int num_shots,
                                         double eps,
                                         curandStatePhilox4_32_10_t* rng,
                                         unsigned int j_start,
                                         unsigned int j_stride)
{
    unsigned int mark = 0;

    for (unsigned int j = j_start; j < num_shots; j += j_stride) {
        double x = x0;
        double y = y0;

        while (true) {
            reflect_inside(x, y);

            double r = find_min_radius_dev(x, y);
            if (r < eps) {
                if (hit_dirichlet_and_is_top(x, y, eps)) ++mark;
                break;
            }

            double u = curand_uniform_double(rng);  // (0,1]
            double theta = 2.0 * PI * u;
            double s = sin(theta), c = cos(theta);
            x += r * s;
            y += r * c;
        }
    }

    return mark;
}

/* --------------------- CUDA kernel: block-per-point with strided shots and shared reduction --------------------- */
__global__ void wos_kernel(const double* __restrict__ xs,
                           const double* __restrict__ ys,
                           const unsigned int* __restrict__ shots,
                           double* __restrict__ results,
                           int n,
                           double eps,
                           unsigned long long seed) {
    int p = blockIdx.x; 
    if (p >= n) return;

    int tid = threadIdx.x;

    double x0 = xs[p];
    double y0 = ys[p];
    unsigned int num_shots = shots[p];

    curandStatePhilox4_32_10_t rng;
    curand_init(seed, (unsigned long long)p * blockDim.x + tid, 0, &rng);

    unsigned int local_mark = wos_walk_strided(x0, y0, num_shots, eps,
                                               &rng,
                                               (unsigned int)tid,
                                               (unsigned int)blockDim.x);

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
        results[p] = (double)sdata[0] / (double)num_shots;
    }
}

/* --------------------- main --------------------- */
int main() {
    time_t rawtime; struct tm* timeinfo;
    time(&rawtime); timeinfo = localtime(&rawtime);
    printf("Current local time and date: %s", asctime(timeinfo));
    printf("Every point has %u shots\n", NUM_SHOTS_PER_PT);

    int num_pts = 0;
    Point* list = get_points(&num_pts);

    const int N = num_pts;

    std::vector<double> hx(N), hy(N), hres(N, 0.0);
    std::vector<unsigned int> hshots(N, NUM_SHOTS_PER_PT);
    #pragma omp parallel for
    for (int i = 0; i < N; ++i) {
        hx[i] = list[i]->x;
        hy[i] = list[i]->y;
        hshots[i] = list[i]->num_shots;
    }

    double *dx = nullptr, *dy = nullptr, *dres = nullptr;
    unsigned int *dshots = nullptr;
    cudaMalloc((void**)&dx,    N * sizeof(double));
    cudaMalloc((void**)&dy,    N * sizeof(double));
    cudaMalloc((void**)&dres,  N * sizeof(double));
    cudaMalloc((void**)&dshots,N * sizeof(unsigned int));

    cudaMemcpy(dx, hx.data(), N * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(dy, hy.data(), N * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(dshots, hshots.data(), N * sizeof(unsigned int), cudaMemcpyHostToDevice);

    const int TPB = 256;
    const int blocks = N;  
    unsigned long long seed = (unsigned long long)time(NULL);
    size_t shmem_size = TPB * sizeof(unsigned int);

    wos_kernel<<<blocks, TPB, shmem_size>>>(dx, dy, dshots, dres, N, EPSILON, seed);
    cudaDeviceSynchronize();

    cudaMemcpy(hres.data(), dres, N * sizeof(double), cudaMemcpyDeviceToHost);
    #pragma omp parallel for
    for (int i = 0; i < N; ++i) {
        list[i]->result = hres[i];
    }


    print_points(list, N);
    destroy_points(list, num_pts);

    cudaFree(dx); cudaFree(dy); cudaFree(dres); cudaFree(dshots);

    time(&rawtime); timeinfo = localtime(&rawtime);
    printf("Finish time and date: %s", asctime(timeinfo));
    return 0;
}
