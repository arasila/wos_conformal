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
        denom = x * x + y * y;
        x = x / denom;
        y = y / denom;
        return false;
    } else if (sqrt(pow(x + 1.16452, 2) + pow(y + 0.482362, 2)) < 0.767327) {
        denom = pow(x + 1.16452, 2) + pow(y + 0.482362, 2);
        x = 0.767327 * 0.767327 * (x + 1.16452) / denom - 1.16452;
        y = 0.767327 * 0.767327 * (y + 0.482362) / denom - 0.482362;
        return false;
    } else if (sqrt(pow(x - 1.0, 2) + pow(y - 0.57735, 2)) < 0.57735) {
        denom = pow(x - 1.0, 2) + pow(y - 0.57735, 2);
        x = 0.57735 * 0.57735 * (x - 1.0) / denom + 1.0;
        y = 0.57735 * 0.57735 * (y - 0.57735) / denom + 0.57735;
        return false;
    }
    return true;
}

// 主模拟函数
double WalkOnSpheres(double x1, double y1, int NN, double eps) {
double wx = x1, wy = y1;
double ux = 0;
int j = 0;
double distance, theta;

// 定义圆的中心和半径
double center2x = -1.16452, center2y = -0.482362, r2 = 0.767327;
double center4x = 1.0, center4y = 0.57735, r4 = 0.57735;
double Tarc2Centerx = 0.861763, Tarc2Centery = 0.509672, Tarc2Radius = 0.0490051;
double Tarc4Centerx = -0.931741, Tarc4Centery = -0.368395, Tarc4Radius = 0.0620912;

	while (j < NN) {
		// 反射处理
		if (euclidean_distance(wx, wy, center2x, center2y) < r2) {
			anti_conformal_transform(center2x, center2y, r2, &wx, &wy);
		} else if (euclidean_distance(wx, wy, center4x, center4y) < r4) {
			anti_conformal_transform(center4x, center4y, r4, &wx, &wy);
		}
			
			// 计算下一个行走的半径
			distance = fmin(
			euclidean_distance(wx, wy, Tarc2Centerx, Tarc2Centery) - Tarc2Radius,
			euclidean_distance(wx, wy, Tarc4Centerx, Tarc4Centery) - Tarc4Radius
		);
		distance = fmin(distance, 1 - sqrt(wx * wx + wy * wy));
		
		// 判断是否到达epsilon-shell
		if (distance < eps) {
			ux += (wy < 0.1) ? 0 : 1;
			wx = x1;
			wy = y1;
			j++;
		} else {
		// 随机选择一个方向并更新位置
			theta = random_double() * 2 * PI;
			wx += distance * cos(theta);
			wy += distance * sin(theta);
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

inline double find_min_radius(double x, double y) {
    double center2x = -1.16452, center2y = -0.482362, r2 = 0.767327;
    double center4x = 1.0, center4y = 0.57735, r4 = 0.57735;
    double dist1 = sqrt(x * x + y * y) - 1;
    double dist2 = sqrt(pow(x - center2x, 2) + pow(y - center2y, 2)) - r2;
    double dist3 = sqrt(pow(x - center4x, 2) + pow(y - center4y, 2)) - r4;
    return fmin(dist1, fmin(dist2, dist3));
}

inline void move_to_next(Point pt, double r) {
    double theta = ((double)(rand())) * 2 * PI / (RAND_MAX);
    pt->x += r * cos(theta);
    pt->y += r * sin(theta);
}
/*
inline void bounce_inside(Point pt) {
    double denom;
    if (sqrt(pt->x * pt->x + pt->y * pt->y) > 1) {
        denom = pt->x * pt->x + pt->y * pt->y;
        pt->x = pt->x / denom;
        pt->y = pt->y / denom;
    } else if (sqrt(pow(pt->x + 1.16452, 2) + pow(pt->y + 0.482362, 2)) < 0.767327) {
        denom = pow(pt->x + 1.16452, 2) + pow(pt->y + 0.482362, 2);
        pt->x = 0.767327 * 0.767327 * (pt->x + 1.16452) / denom - 1.16452;
        pt->y = 0.767327 * 0.767327 * (pt->y + 0.482362) / denom - 0.482362;
    } else if (sqrt(pow(pt->x - 1.0, 2) + pow(pt->y - 0.57735, 2)) < 0.57735) {
        denom = pow(pt->x - 1.0, 2) + pow(pt->y - 0.57735, 2);
        pt->x = 0.57735 * 0.57735 * (pt->x - 1.0) / denom + 1.0;
        pt->y = 0.57735 * 0.57735 * (pt->y - 0.57735) / denom + 0.57735;
    }
}*/

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
