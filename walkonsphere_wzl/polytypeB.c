#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <math.h>
#include <omp.h>
#include <time.h>
#include <string.h>

typedef struct {
    double x;
    double y;
    double result;
    unsigned int num_shots;
} point_t, *Point;


/* slide the volumn to select points  */
#define X_SLIDES 120 
#define Y_SLIDES 100
#define NUM_POINTS X_SLIDES * Y_SLIDES
#define NUM_SHOTS 1000.0 /* Must be a float number */
#define PI 3.14159265358979323846
#define EPSILON 1e-6
#define length(x, y) sqrt((x)*(x)+(y)*(y))
inline Point create_point(double, double);
inline Point copy_point(Point);
inline Point* get_points(int*);
inline void destroy_points(Point*, int);
inline void print_points(Point*, int, char*);
inline double find_min_radius(double, double);
inline void move_to_next(Point, double);
// 生成[0,1)之间的随机数
double random_double() {
return (double)rand() / (double)RAND_MAX;
}

// 计算两点之间的欧几里德距离
double euclidean_distance(double x1, double y1, double x2, double y2) {
return sqrt(pow(x1 - x2, 2) + pow(y1 - y2, 2));
}

// 反射映射函数
void anti_conformal_transform(double x1, double y1, double r, double *wx, double *wy) {
double denom = pow(*wx - x1, 2) + pow(*wy - y1, 2);
*wx = r * r * (*wx - x1) / denom + x1;
*wy = r * r * (*wy - y1) / denom + y1;
}
inline bool isOnDirichlet(Point, int*);
inline double walk_on_spheres(double x, double y, unsigned int num_shots, double epsilon); 
inline bool bounce_inside(double x, double y) {
    double denom;
    if (sqrt(x * x + y * y) > 1) {
        return false;
    }
    return true;
}typedef struct {
double re;
double im;
} Complex;



// 反射球面行走算法
double WalkOnSpheres(double x1, double x2, int NN, double eps) {
	int m = 2, n = 10, r = 12;
	Complex v[4];
	
	// 计算四个顶点
	v[0].re = cos(m * PI / 24);
	v[0].im = sin(m * PI / 24);
	v[1].re = cos(n * PI / 24);
	v[1].im = sin(n * PI / 24);
	v[2].re = cos(r * PI / 24);
	v[2].im = sin(r * PI / 24);
	v[3].re = 1.0;
	v[3].im = 0.0;
	
	double wx = x1, wy = x2, ux = 0;
	int j = 0;
	double distance, theta;
	
	while (j < NN) {
	// 如果粒子在圆外，则将其反射回圆内
	if (wx * wx + wy * wy > 1) {
		double norm = sqrt(wx * wx + wy * wy);
		wx /= norm;
		wy /= norm;
	}
	
	// 计算下一次行走的距离
	if (wx >= 0 && wy >= 0 && !(tan(2 * PI / 24) < (wy / wx) && (wy / wx) < tan(10 * PI / 24))) {
		distance = 1 - sqrt(wx * wx + wy * wy);
	} else {
		double d1 = sqrt((wx - 1) * (wx - 1) + wy * wy);
		double d2 = sqrt((wx - v[0].re) * (wx - v[0].re) + (wy - v[0].im) * (wy - v[0].im));
		double d3 = sqrt((wx - v[1].re) * (wx - v[1].re) + (wy - v[1].im) * (wy - v[1].im));
		double d4 = sqrt((wx - v[2].re) * (wx - v[2].re) + (wy - v[2].im) * (wy - v[2].im));
		distance = fmin(fmin(d1, d2), fmin(d3, d4));
	}
	
	if (distance < eps) {
		ux += (wx < 0.5) ? 1 : 0;
		wx = x1;
		wy = x2;
		j++;
	} else {
		theta = random_double();
		wx += distance * cos(theta * 2 * PI);
		wy += distance * sin(theta * 2 * PI);
	}
}

return ux / NN;
}


int main(int argc, char** argv) {

    time_t rawtime;
    struct tm * timeinfo;
    time(&rawtime);
    timeinfo = localtime(&rawtime);
    printf("Current local time and date: %s", asctime(timeinfo));
    printf("Every point has %lf shots\n", NUM_SHOTS);
    // Get a list of points
    int num_of_pts;
    Point* list = get_points(&num_of_pts);
    srand(time(NULL));
    int total_section = 1;
    int section = 0;
    int part = num_of_pts / (total_section);
    int start = section * part;
    int end = start + part;
    end = end > num_of_pts ? num_of_pts : end;
	int i;
    #pragma omp parallel for schedule(dynamic)
    for (i = start; i < end; i++) {
        double x = list[i]->x;
        double y = list[i]->y;
        unsigned int num_shots = list[i]->num_shots;
        list[i]->result = WalkOnSpheres(x, y, num_shots, EPSILON);
    }
    print_points(list + start, end - start, "");
    destroy_points(list, num_of_pts);
    time(&rawtime);
    timeinfo = localtime(&rawtime);
    return 0;
}
inline Point copy_point(Point pt) {
    Point new = create_point(pt->x, pt->y);
    new->result = pt->result;
    return new;
}

inline Point create_point(double x, double y) {
    Point new = malloc(sizeof(*new));
    new->x = x;
    new->y = y;
    new->result = 0;
    return new;
}

inline Point* get_points(int* num_of_pts) {
    int counter = 0;
    double x_d = 2.0 / X_SLIDES;
    double y_d = 2.0 / Y_SLIDES;
    int list_Size = NUM_POINTS;
    Point* list = malloc(sizeof(*list) * NUM_POINTS);
    if (!list) {
        perror("Failed to allocate memory");
        exit(EXIT_FAILURE);
    }
    int i;
    int j;
    for (i = 0; i < list_Size; i++) list[i] = NULL;
    for (i = 0; i <= X_SLIDES; i++) {
        double x = -1 + i * x_d;
        for (j = 0; j <= Y_SLIDES; j++) {
            double y = -1 + j * y_d;
            // check validity
            if (!bounce_inside(x, y)) continue; // The point is not in the domain
            list[counter] = create_point(x, y);
            if (y > 0.5 || y < -0.5) list[counter]->num_shots = NUM_SHOTS * 3;
            else list[counter]->num_shots = NUM_SHOTS;
            counter++;
        }
    }
    *num_of_pts = counter;
    return list;
}

inline void destroy_points(Point* list, int num_of_pts) {
	int i;
    for (i = 0; i < num_of_pts; i++)
        free(list[i]);
    free(list);
}

inline void print_points(Point* list, int num_of_pts, char* str) {
    char name[15] = "points";
    strcat(name, str);
    strcat(name, ".txt");
    FILE* fd = fopen(name, "w");
    int i;
    for (i = 0; i < num_of_pts; i++) {
        Point pt = list[i];
        fprintf(fd, "%lf, %lf, %lf\n", pt->x, pt->y, pt->result);
    }
    fclose(fd);
}



inline void move_to_next(Point pt, double r) {
    double theta = ((double)(rand())) * 2 * PI / (RAND_MAX);
    pt->x += r * cos(theta);
    pt->y += r * sin(theta);
}


inline bool isOnDirichlet(Point pt, int* mark) {
    if (1 - pt->y < EPSILON) {
        (*mark)++;
        return true;
    }
    else if (pt->y < EPSILON) {
        return true;
    }
    return false;
}
