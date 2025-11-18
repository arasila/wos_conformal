// source.cu
// CUDA version of source.c (Walk-on-Spheres Monte Carlo)
// Build example: nvcc -O3 -arch=sm_89 source.cu -o source -lcurand

#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <math.h>
#include <time.h>
#include <string.h>

#include <cuda_runtime.h>
#include <curand_kernel.h>

typedef struct {
    double x;
    double y;
    double result;
    unsigned int num_shots;
} point_t, *Point;

/* slide the volume to select points  */
#define Y_SLIDES 100
#define X_SLIDES 100
#define NUM_POINTS ((X_SLIDES + 1) * (Y_SLIDES + 1))

#define NUM_SHOTS 1.0E6 /* Must be a float number */
#define PI 3.14159265358979323846
#define EPSILON 1e-6
#define length(x, y) sqrt((x)*(x)+(y)*(y))

#define v1_x 0.96592582628
#define v1_y 0.25881904510
#define v2_x 0.25881904510
#define v2_y 0.96592582628
#define v3_x 0.0
#define v3_y 1.0
#define v4_x 1.0
#define v4_y 0.0

#define TAN_A 0.2679491924311227   // tan(pi/12)  ~ tan(2/24*pi)
#define TAN_B 3.732050807568877    // tan(5*pi/12) ~ tan(10/24*pi)

#define CUDA_BLOCK_SIZE 256

inline Point create_point(double, double);
inline Point copy_point(Point);
inline Point* get_points(int*);
inline void destroy_points(Point*, int);
inline void print_points(Point*, int);

inline bool inside_domain(double x, double y) {
    if (x * x + y * y > 1.0) return false;
    return true;
}

/* -------------------- Device-side single shot (one Monte Carlo walk) -------------------- */

__device__ int walk_on_spheres_single_shot(double x1, double y1,
                                           curandStatePhilox4_32_10_t* state) {
    double wx = x1;
    double wy = y1;

    while (true) {
        double denom = wx * wx + wy * wy;
        if (denom > 1.0) {
            wx /= denom;
            wy /= denom;
        }

        double distance;
        if (wx >= 0.0 && wy >= 0.0 &&
            !(TAN_A * wx < wy && wy < wx * TAN_B)) {
            // Hit circular boundary region
            distance = 1.0 - sqrt(denom);
        } else {
            // Distance to nearest of the four "corner" points
            double d1 = (wx - v4_x) * (wx - v4_x) + (wy - v4_y) * (wy - v4_y);
            double d2 = (wx - v1_x) * (wx - v1_x) + (wy - v1_y) * (wy - v1_y);
            double d3 = (wx - v2_x) * (wx - v2_x) + (wy - v2_y) * (wy - v2_y);
            double d4 = (wx - v3_x) * (wx - v3_x) + (wy - v3_y) * (wy - v3_y);
            double mind = fmin(d1, fmin(d2, fmin(d3, d4)));
            distance = sqrt(mind);
        }

        if (distance < EPSILON) {
            // Reached boundary, evaluate boundary condition
            return (wx < 0.5) ? 0 : 1;
        } else {
            // Take a random step on the circle
            double u = curand_uniform_double(state);  // (0,1]
            double theta = u * 2.0 * PI;
            wx += distance * cos(theta);
            wy += distance * sin(theta);
        }
    }
}

/* -------------------- CUDA kernel: block-per-point, threads-per-samples -------------------- */

__global__ void walk_on_spheres_kernel(const double* __restrict__ x_arr,
                                       const double* __restrict__ y_arr,
                                       const unsigned int* __restrict__ num_shots_arr,
                                       double* __restrict__ result_arr,
                                       int start_index,
                                       int end_index,
                                       unsigned long long seed) {
    int point_idx = start_index + blockIdx.x;
    if (point_idx >= end_index) return;

    unsigned int total_shots = num_shots_arr[point_idx];
    if (total_shots == 0) {
        if (threadIdx.x == 0) {
            result_arr[point_idx] = 0.0;
        }
        return;
    }

    int tid = threadIdx.x;
    int nthreads = blockDim.x;

    double x0 = x_arr[point_idx];
    double y0 = y_arr[point_idx];

    unsigned int shots_per_thread = (total_shots + nthreads - 1) / nthreads;
    unsigned int start_shot = tid * shots_per_thread;
    if (start_shot >= total_shots) {
        return;
    }
    unsigned int end_shot = start_shot + shots_per_thread;
    if (end_shot > total_shots) end_shot = total_shots;

    unsigned long long gid = (unsigned long long)(point_idx - start_index) * nthreads + (unsigned long long)tid;
    curandStatePhilox4_32_10_t state;
    curand_init(seed, gid, 0ULL, &state);

    unsigned int local_hits = 0U;

    for (unsigned int s = start_shot; s < end_shot; ++s) {
        (void)s; 
        local_hits += (unsigned int)walk_on_spheres_single_shot(x0, y0, &state);
    }

    __shared__ unsigned long long sh_hits[CUDA_BLOCK_SIZE];
    sh_hits[tid] = (unsigned long long)local_hits;
    __syncthreads();

    for (int offset = nthreads / 2; offset > 0; offset >>= 1) {
        if (tid < offset) {
            sh_hits[tid] += sh_hits[tid + offset];
        }
        __syncthreads();
    }

    if (tid == 0) {
        double hits = (double)sh_hits[0];
        result_arr[point_idx] = hits / (double)total_shots;
    }
}

/* -------------------- Host utilities (mostly unchanged) -------------------- */

void random_permute(Point* array, int length) {
    srand((unsigned int)time(NULL));
    for (int i = length - 1; i > 0; i--) {
        int j = rand() % (i + 1);
        Point temp = array[i];
        array[i] = array[j];
        array[j] = temp;
    }
}

inline Point copy_point(Point pt) {
    Point new_pt = create_point(pt->x, pt->y);
    new_pt->result = pt->result;
    new_pt->num_shots = pt->num_shots;
    return new_pt;
}

inline Point create_point(double x, double y) {
    Point new_pt = (Point)malloc(sizeof(*new_pt));
    if (!new_pt) {
        perror("Failed to allocate point");
        exit(EXIT_FAILURE);
    }
    new_pt->x = x;
    new_pt->y = y;
    new_pt->result = 0.0;
    new_pt->num_shots = 0;
    return new_pt;
}

inline Point* get_points(int* num_of_pts) {
    int counter = 0;
    double x_d = 2.0 / X_SLIDES;
    double y_d = 2.0 / Y_SLIDES;
    double x;
    double y;
    int list_Size = NUM_POINTS;
    Point* list = (Point*)malloc(sizeof(*list) * NUM_POINTS);
    if (!list) {
        perror("Failed to allocate memory");
        exit(EXIT_FAILURE);
    }

    for (int i = 0; i < list_Size; i++) list[i] = NULL;

    for (int i = 0; i <= X_SLIDES; i++) {
        x = -1.0 + i * x_d;
        for (int j = 0; j <= Y_SLIDES; j++) {
            y = -1.0 + j * y_d;
            if (!inside_domain(x, y)) continue; // The point is not in the domain
            list[counter] = create_point(x, y);
            list[counter]->num_shots = (unsigned int)NUM_SHOTS;
            counter++;
        }
    }

    // Smoothen the boundary
    for (int i = 0; i <= 500; i++) {
        for (int j = 0; j < 3; j++) {
            if (counter >= list_Size) {
                list_Size *= 2; // Double the size of the list
                list = (Point*)realloc(list, sizeof(*list) * list_Size);
                if (!list) {
                    perror("Failed to reallocate memory");
                    exit(EXIT_FAILURE);
                }
            }
            double r = 0.95 + 0.025 * j;
            double dtheta = 2.0 * PI / 500.0;
            double t = i * dtheta;
            x = r * cos(t);
            y = r * sin(t);
            list[counter] = create_point(x, y);
            list[counter]->num_shots = (unsigned int)NUM_SHOTS;
            counter++;
        }
    }
    *num_of_pts = counter;
    return list;
}

inline void destroy_points(Point* list, int num_of_pts) {
    for (int i = 0; i < num_of_pts; i++) {
        free(list[i]);
    }
    free(list);
}

inline void print_points(Point* list, int num_of_pts) {
    const char* filename = "points.txt";
    FILE* fd = fopen(filename, "w");
    if (!fd) {
        perror("Failed to open output file");
        return;
    }
    for (int i = 0; i < num_of_pts; i++) {
        Point pt = list[i];
        fprintf(fd, "%lf, %lf, %lf\n", pt->x, pt->y, pt->result);
    }
    fclose(fd);
}

/* -------------------- Main: same interface, GPU compute -------------------- */

int main(void) {

    time_t rawtime;
    struct tm* timeinfo;
    time(&rawtime);
    timeinfo = localtime(&rawtime);
    printf("Current local time and date: %s", asctime(timeinfo));
    printf("Every point has %lf shots\n", NUM_SHOTS);

    // Get a list of points
    int num_of_pts;
    Point* list = get_points(&num_of_pts);
    random_permute(list, num_of_pts);

    int start = 0;
    int end = num_of_pts;
    printf("Processing all %d points\n", num_of_pts);

    // -------------------- Build SoA arrays for GPU --------------------
    double* h_x = (double*)malloc(sizeof(double) * num_of_pts);
    double* h_y = (double*)malloc(sizeof(double) * num_of_pts);
    unsigned int* h_num_shots = (unsigned int*)malloc(sizeof(unsigned int) * num_of_pts);
    double* h_result = (double*)malloc(sizeof(double) * num_of_pts);

    if (!h_x || !h_y || !h_num_shots || !h_result) {
        perror("Failed to allocate host arrays");
        destroy_points(list, num_of_pts);
        return EXIT_FAILURE;
    }

    for (int i = 0; i < num_of_pts; i++) {
        h_x[i] = list[i]->x;
        h_y[i] = list[i]->y;
        h_num_shots[i] = list[i]->num_shots;
        h_result[i] = 0.0;
    }

    // -------------------- Allocate device memory --------------------
    double* d_x = NULL;
    double* d_y = NULL;
    unsigned int* d_num_shots = NULL;
    double* d_result = NULL;

    cudaMalloc((void**)&d_x, sizeof(double) * num_of_pts);
    cudaMalloc((void**)&d_y, sizeof(double) * num_of_pts);
    cudaMalloc((void**)&d_num_shots, sizeof(unsigned int) * num_of_pts);
    cudaMalloc((void**)&d_result, sizeof(double) * num_of_pts);

    cudaMemcpy(d_x, h_x, sizeof(double) * num_of_pts, cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, h_y, sizeof(double) * num_of_pts, cudaMemcpyHostToDevice);
    cudaMemcpy(d_num_shots, h_num_shots, sizeof(unsigned int) * num_of_pts, cudaMemcpyHostToDevice);
    cudaMemcpy(d_result, h_result, sizeof(double) * num_of_pts, cudaMemcpyHostToDevice);

    // -------------------- Launch kernel (block-per-point) --------------------
    int points_in_section = end - start;
    dim3 grid(points_in_section);
    dim3 block(CUDA_BLOCK_SIZE);

    unsigned long long seed = (unsigned long long)time(NULL);

    walk_on_spheres_kernel<<<grid, block>>>(
        d_x, d_y, d_num_shots, d_result,
        start, end, seed
    );
    cudaDeviceSynchronize();

    // -------------------- Copy result back and write into list --------------------
    cudaMemcpy(h_result, d_result, sizeof(double) * num_of_pts, cudaMemcpyDeviceToHost);

    for (int i = start; i < end; i++) {
        list[i]->result = h_result[i];
    }

    print_points(list + start, end - start);

    // -------------------- Cleanup --------------------
    cudaFree(d_x);
    cudaFree(d_y);
    cudaFree(d_num_shots);
    cudaFree(d_result);

    free(h_x);
    free(h_y);
    free(h_num_shots);
    free(h_result);

    destroy_points(list, num_of_pts);

    time(&rawtime);
    timeinfo = localtime(&rawtime);

    return 0;
}
