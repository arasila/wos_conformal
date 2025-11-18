// Build: nvcc -O3 -arch=sm_89 --use_fast_math source.cu -o source -lcurand

#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <math.h>
#include <time.h>
#include <string.h>

typedef struct {
    float x;
    float y;
    float result;
    unsigned int num_shots;
} point_t, *Point;

/* grid definition (unchanged) */
#define X_SLIDES 120
#define Y_SLIDES 100
#define NUM_POINTS (X_SLIDES * Y_SLIDES)
#define NUM_SHOTS 50000000.0f
#define PI 3.14159265358979323846f
#define EPSILON 1e-6f

/* ---------------- Host helpers: keep logic identical ---------------- */
static inline Point create_point(float x, float y) {
    Point p = (Point)malloc(sizeof(*p));
    if (!p) { perror("malloc Point"); exit(EXIT_FAILURE); }
    p->x = x; p->y = y; p->result = 0.0f; p->num_shots = 0u;
    return p;
}

static inline Point* get_points(int* num_of_pts) {
    int counter = 0;
    float x_d = 3.0f / (float)X_SLIDES;
    float y_d = 2.0f / (float)Y_SLIDES;
    int list_Size = NUM_POINTS;

    Point* list = (Point*)malloc(sizeof(*list) * NUM_POINTS);
    if (!list) { perror("Failed to allocate list"); exit(EXIT_FAILURE); }
    for (int i = 0; i < list_Size; i++) list[i] = NULL;

    // base grid
    for (int i = 0; i <= X_SLIDES; i++) {
        float x = i * x_d;
        for (int j = 0; j <= Y_SLIDES; j++) {
            float y = j * y_d;
            if (x > 2.f && y > 1.f) continue;
            list[counter] = create_point(x, y);
            if (y > 1.5f || y < 0.5f) list[counter]->num_shots = (unsigned int)(NUM_SHOTS * 3.f);
            else                      list[counter]->num_shots = (unsigned int)(NUM_SHOTS);
            counter++;
        }
    }

    // densify near (2,1) — keep exactly
    for (int i = 0; i <= 20; i++){
        float x = 1.9f + (i + 1) * 0.01f;
        for (int j = 0; j <= 20; j++){
            float y = 0.95f + (j + 1) * 0.0101f;
            if (x > 2.f && y > 1.f) continue;
            if (counter >= list_Size) {
                list_Size *= 2;
                list = (Point*)realloc(list, sizeof(*list) * list_Size);
                if (!list) { perror("realloc list"); exit(EXIT_FAILURE); }
            }
            list[counter] = create_point(x, y);
            list[counter]->num_shots = (unsigned int)(NUM_SHOTS);
            counter++;
        }
    }

    *num_of_pts = counter;
    return list;
}

static inline void destroy_points(Point* list, int num_of_pts) {
    for (int i = 0; i < num_of_pts; i++) free(list[i]);
    free(list);
}

static inline void print_points(Point* list, int num_of_pts) {
    FILE* fd = fopen("points.txt", "w");
    if (!fd) { perror("fopen output"); return; }
    for (int i = 0; i < num_of_pts; i++) {
        Point pt = list[i];
        fprintf(fd, "%f, %f, %f\n", pt->x, pt->y, pt->result);
    }
    fclose(fd);
}

static inline void random_permute(Point *array, int length) {
    srand((unsigned)time(NULL));
    for (int i = length - 1; i > 0; i--) {
        int j = rand() % (i + 1);
        Point tmp = array[i]; array[i] = array[j]; array[j] = tmp;
    }
}

/* ---------------- Device helpers ---------------- */
__device__ inline float d_length(float x, float y) { return sqrtf(x*x + y*y); }

/* One block per point. Threads split shots; reduction in shared memory. */
__global__ void wos_block_per_point(const float* __restrict__ xs,
                                    const float* __restrict__ ys,
                                    const unsigned int* __restrict__ shots,
                                    float* __restrict__ results,
                                    int n, unsigned long long seed)
{
    int p = blockIdx.x;           // point index
    if (p >= n) return;

    float x1 = xs[p];
    float y1 = ys[p];
    unsigned int num_shots = shots[p];

    // per-thread RNG, unique sequence per (point, thread)
    int tid = threadIdx.x;
    curandStatePhilox4_32_10_t rng;
    curand_init(seed, (unsigned long long)p * blockDim.x + tid, 0ULL, &rng);

    unsigned int local_mark = 0;

    // strided over shots
    for (unsigned int j = tid; j < num_shots; j += blockDim.x) {
        // start a shot from (x1,y1), walk until hit (r < EPSILON)
        float wx = x1, wy = y1;
        bool done = false;
        while (!done) {
            // domain mapping (unchanged)
            if (wy < 0.f) {
                wy = -wy;
            } else if (wy > 4.f) {
                wy = wy - 4.f;
            } else if (wy > 1.f && wy < 3.f && wx > 2.f) {
                wx = 4.f - wx;
            }

            float dist1 = wx;
            float dist2 = d_length(wx - 2.f, wy - 1.f);
            float dist3 = d_length(wx - 2.f, wy - 3.f);
            float r;
            if (wx <= 2.f) {
                r = fminf(dist1, fminf(dist2, dist3));
            } else {
                if (wy < 1.f)       r = fminf(1.f - wy, 3.f - wx);
                else if (wy > 3.f)  r = fminf(3.f - wx, wy - 3.f);
                else                r = fminf(wy - 1.f, 3.f - wy);
            }

            if (r < EPSILON) {
                if (wx > 0.1f) local_mark++;
                done = true;
                break;
            }

            // random direction
            float u = curand_uniform(&rng); // (0,1]
            float ang = u * (2.f * PI);
            float s = __sinf(ang), c = __cosf(ang);
            wx += r * s;
            wy += r * c;
        }
    }

    // reduction in shared memory
    extern __shared__ unsigned int sdata[];
    sdata[tid] = local_mark;
    __syncthreads();

    // tree reduction
    for (int offset = blockDim.x >> 1; offset > 0; offset >>= 1) {
        if (tid < offset) sdata[tid] += sdata[tid + offset];
        __syncthreads();
    }

    if (tid == 0) {
        unsigned int total_mark = sdata[0];
        results[p] = (float)total_mark / (float)num_shots;
    }
}

int main() {

    time_t rawtime; struct tm* timeinfo;
    time(&rawtime); timeinfo = localtime(&rawtime);
    printf("Current local time and date: %s", asctime(timeinfo));
    printf("Every point has %.1f shots\n", NUM_SHOTS);

    int num_of_pts;
    Point* list = get_points(&num_of_pts);
    random_permute(list, num_of_pts); // preserve your behavior

    int n = num_of_pts;
    if (n <= 0) {
        fprintf(stderr, "Empty section (n<=0). Nothing to do.\n");
        destroy_points(list, num_of_pts);
        return 0;
    }

    // Pack SoA
    float *hx = (float*)malloc(n * sizeof(float));
    float *hy = (float*)malloc(n * sizeof(float));
    unsigned int *hshots = (unsigned int*)malloc(n * sizeof(unsigned int));
    float *hres = (float*)malloc(n * sizeof(float));
    if (!hx || !hy || !hshots || !hres) { perror("malloc host buffers"); exit(EXIT_FAILURE); }

    for (int i = 0; i < n; ++i) {
        hx[i]     = list[i]->x;
        hy[i]     = list[i]->y;
        hshots[i] = list[i]->num_shots;
        hres[i]   = 0.0f;
    }

    // Device buffers
    float *dx = nullptr, *dy = nullptr, *dres = nullptr;
    unsigned int *dshots = nullptr;
    cudaMalloc((void**)&dx, n * sizeof(float));
    cudaMalloc((void**)&dy, n * sizeof(float));
    cudaMalloc((void**)&dshots, n * sizeof(unsigned int));
    cudaMalloc((void**)&dres, n * sizeof(float));

    cudaMemcpy(dx, hx, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dy, hy, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dshots, hshots, n * sizeof(unsigned int), cudaMemcpyHostToDevice);

    // Launch: one block per point, 256 threads each; dynamic shared mem for reduction
    const int threads = 256;
    dim3 block(threads);
    dim3 grid(n); // block-per-point
    size_t shmem = threads * sizeof(unsigned int);
    unsigned long long seed = (unsigned long long)time(NULL);

    wos_block_per_point<<<grid, block, shmem>>>(dx, dy, dshots, dres, n, seed);
    cudaDeviceSynchronize();

    cudaMemcpy(hres, dres, n * sizeof(float), cudaMemcpyDeviceToHost);

    // write back & output
    for (int i = 0; i < n; ++i) list[i]->result = hres[i];
    print_points(list, n);

    cudaFree(dx); cudaFree(dy); cudaFree(dshots); cudaFree(dres);
    free(hx); free(hy); free(hshots); free(hres);
    destroy_points(list, num_of_pts);
    return 0;
}
