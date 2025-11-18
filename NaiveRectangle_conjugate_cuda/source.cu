// source.cu
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

typedef struct {
    double x;
    double y;
    double result;
    unsigned int num_shots;
} point_t, *Point;


//Define the height of the rectangle
#define HEIGHT 1.0
/* slide the volumn to select points  */
#define X_SLIDES 10 
#define Y_SLIDES 10
//#define Z_SLIDES 20
#define NUM_POINTS (X_SLIDES + 1) * (Y_SLIDES + 1)
#define NUM_SHOTS 1.0E6 /* Must be a float number */
#define PI 3.14159265358979323846
#define EPSILON 1e-6
//#define TEST
#define length(x, y) sqrt((x)*(x)+(y)*(y))

inline Point create_point(double, double);
inline Point copy_point(Point);
inline Point* get_points(int*);
inline void destroy_points(Point*, int);
inline void print_points(Point*, int);
inline double walk_on_spheres(double x, double y, unsigned int num_shots, double epsilon);


__device__ double walk_on_spheres_device(double x1, double y1,
                                         unsigned int num_shots,
                                         double epsilon,
                                         curandStatePhilox4_32_10_t *state) {
    double wx = x1;
    double wy = y1;
    unsigned int mark = 0;
    unsigned int j = 0;

    while (j < num_shots) {
        if (wx < 0.0) {
            wx = -wx;
        } else if (wx > 1.0) {
            wx = 2.0 - wx;
        }
        double r = (HEIGHT - wy) > wy ? wy : (HEIGHT - wy);
        if (r < EPSILON) {
            if (wy > HEIGHT / 2.0) mark++;
            wx = x1;
            wy = y1;
            j++;
        }
        double u1 = curand_uniform_double(state) * 2.0 * PI;
        double z1 = sin(u1);
        double z2 = cos(u1);
        wx += r * z1;
        wy += r * z2;
    }
    return ((double)mark) / (double)num_shots;
}

/* ---------------- kernel ---------------- */

__global__ void wos_kernel(const double *x,
                           const double *y,
                           const unsigned int *num_shots,
                           double *result,
                           int n,
                           double epsilon,
                           unsigned long long seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    curandStatePhilox4_32_10_t state;
    curand_init(seed, idx, 0, &state);

    double val = walk_on_spheres_device(x[idx], y[idx],
                                        num_shots[idx],
                                        epsilon,
                                        &state);
    result[idx] = val;
}


int main() {
    time_t rawtime;
    struct tm * timeinfo;
    time(&rawtime);
    timeinfo = localtime(&rawtime);
    printf("Current local time and date: %s", asctime(timeinfo));
    printf("Every point has %lf shots\n", NUM_SHOTS);

    //Get a list of points
    int num_of_pts;
    Point* list = get_points(&num_of_pts);
    srand(time(NULL));

    int start = 0;
    int end   = num_of_pts;
    int n     = end - start;
    if (n > 0) {
        // host buffers (SoA)
        double *h_x   = (double*)malloc(n * sizeof(double));
        double *h_y   = (double*)malloc(n * sizeof(double));
        unsigned int *h_num_shots = (unsigned int*)malloc(n * sizeof(unsigned int));
        double *h_result = (double*)malloc(n * sizeof(double));

        for (int k = 0; k < n; ++k) {
            Point pt = list[start + k];
            h_x[k] = pt->x;
            h_y[k] = pt->y;
            h_num_shots[k] = pt->num_shots;
            h_result[k] = 0.0;
        }

        // device buffers
        double *d_x = NULL, *d_y = NULL, *d_result = NULL;
        unsigned int *d_num_shots = NULL;

        cudaMalloc((void**)&d_x,         n * sizeof(double));
        cudaMalloc((void**)&d_y,         n * sizeof(double));
        cudaMalloc((void**)&d_num_shots, n * sizeof(unsigned int));
        cudaMalloc((void**)&d_result,    n * sizeof(double));

        cudaMemcpy(d_x,         h_x,         n * sizeof(double),       cudaMemcpyHostToDevice);
        cudaMemcpy(d_y,         h_y,         n * sizeof(double),       cudaMemcpyHostToDevice);
        cudaMemcpy(d_num_shots, h_num_shots, n * sizeof(unsigned int), cudaMemcpyHostToDevice);

        int threads = 256;
        int blocks  = (n + threads - 1) / threads;
        unsigned long long seed = (unsigned long long)time(NULL);

        wos_kernel<<<blocks, threads>>>(d_x, d_y, d_num_shots, d_result,
                                        n, EPSILON, seed);
        cudaDeviceSynchronize();

        cudaMemcpy(h_result, d_result, n * sizeof(double), cudaMemcpyDeviceToHost);

        for (int k = 0; k < n; ++k) {
            list[start + k]->result = h_result[k];
        }

        cudaFree(d_x);
        cudaFree(d_y);
        cudaFree(d_num_shots);
        cudaFree(d_result);

        free(h_x);
        free(h_y);
        free(h_num_shots);
        free(h_result);
    }

    print_points(list + start, end - start);
    destroy_points(list, num_of_pts);
    time(&rawtime);
    timeinfo = localtime(&rawtime);
#ifdef TEST
    printf("Finish time and date: %s\n", asctime(timeinfo));
#endif
    return 0;
}


inline double walk_on_spheres(double x1, double y1, unsigned int num_shots, double epsilon){
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
    return ((double)mark)/num_shots;
}


inline Point copy_point(Point pt) {
    Point newp = create_point(pt->x, pt->y);
    newp->result = pt->result;
    return newp;
}

inline Point create_point(double x, double y) {
    Point newp = (Point)malloc(sizeof(*newp));
    newp->x = x;
    newp->y = y;
    newp->result = 0;
    newp->num_shots = 0;
    return newp;
}
    
inline Point* get_points(int* num_of_pts) {
    int counter = 0;
    double x_d = 1.0/X_SLIDES;
    double y_d = 1.0/Y_SLIDES;
    int list_Size = NUM_POINTS;
    Point* list = (Point*)malloc(sizeof(*list) * NUM_POINTS);
    if (!list) {
        perror("Failed to allocate memory");
        exit(EXIT_FAILURE);
    }
    for (int i = 0; i < list_Size; i++) list[i] = NULL;
    for (int i = 0; i <= X_SLIDES; i++) {
        double x = i * x_d;
        printf("i = %d\n", i);
        for (int j = 0; j <= Y_SLIDES; j++) {
            double y = j * y_d;
            if (counter >= list_Size){
                list_Size *= 2; // Double the size of the list
                list = (Point*)realloc(list, sizeof(*list) * list_Size);
                if (!list) {
                    perror("Failed to reallocate memory");
                    exit(EXIT_FAILURE);
                }
            }
            list[counter] = create_point(x, y);
            list[counter]->num_shots = NUM_SHOTS;
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
    FILE* fd = fopen("points.txt", "w");
    if (!fd) {
        perror("fopen points.txt");
        return;
    }
    for (int i = 0; i < num_of_pts; i++) {
        Point pt = list[i];
        fprintf(fd, "%lf, %lf, %lf\n", pt->x, pt->y, pt->result);
    }
    fclose(fd);
}
