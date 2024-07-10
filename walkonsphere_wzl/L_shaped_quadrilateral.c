#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <math.h>
#include <omp.h>
#include <time.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>


#define PI 3.14159265358979323846

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

// 主模拟函数
double WalkOnSpheresPolyTypeA(double x1, double y1, int NN, double eps) {
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
typedef struct {
	double x;
	double y;
	double result;
	unsigned int num_shots;
} point_t, *Point;

/* slide the volumn to select points  */
#define X_SLIDES 120 
#define Y_SLIDES 120
//#define Z_SLIDES 20
#define NUM_POINTS X_SLIDES * Y_SLIDES
#define NUM_SHOTS 10000.0 /* Must be a float number */
#define PI 3.14159265358979323846
#define EPSILON 1e-6
//#define TEST
#define length(x, y) sqrt((x)*(x)+(y)*(y))
inline Point create_point(double, double);
inline Point copy_point(Point);
inline Point* get_points(int*);
inline void destroy_points(Point*, int);
inline void print_points(Point*, int, char*);
inline double find_min_radius(double, double);
inline void move_to_next(Point, double);
inline void bounce_inside(Point);
inline bool isOnDirichlet(Point, int*);
inline double walk_on_spheres(double x, double y, unsigned int num_shots, double epsilon); 



// 算法核心函数
double WalkOnSpheres(double x1, double x2, int NN, double eps) {
    double wx = x1;
    double wy = x2;
    double ux = 0;
    int j = 0;
    double distance, theta;

    while (j < NN) {
        // 反射处理
        if (wx < 0) {
            wx = -wx;
        } else if (wx > 6) {
            wx = 12 - wx;
        }

        if (wy < 0) {
            wy = -wy;
        } else if (wy > 4) {
            wy = 8 - wy;
        }

        // 计算下一个行走的半径
        if (wy <= 1) {
            distance = fmin(wy, fmin(sqrt(pow(wx - 2, 2) + pow(wy - 1, 2)), sqrt(pow(wx - 4, 2) + pow(wy - 1, 2))));
        } else if (wx <= 2) {
            distance = fmin(2 - wy, 2 - wx);
        } else if (wx > 4) {
            distance = fmin(2 - wy, wx - 4);
        } else {
            distance = fmin(wx - 2, 4 - wx);
        }

        // 判断是否到达epsilon-shell
        if (distance < eps) {
            ux += (wy < 0.1) ? 0 : 1;
            wx = x1;
            wy = x2;
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
  	time( &rawtime );
  	timeinfo = localtime( &rawtime );
  	printf( "Current local time and date: %s", asctime (timeinfo) );
	printf( "Every point has %lf shots\n", NUM_SHOTS);
	//Get a list of points
	int num_of_pts;
	Point* list = get_points(&num_of_pts);
	srand(time(NULL));
//	omp_set_num_threads(100);
	int total_section = 1;
	int section = 0;
	int part = num_of_pts/(total_section);
	int start = section * part;
	int end = start + part;
	end = end>num_of_pts?num_of_pts:end;
#ifdef TEST
	printf("This job is from %d to %d\n", start+1, end);
#endif
//	for (int i = start; i < end; i++) {
//		printf("%lf, %lf, %lf\n", list[i]->x, list[i]->y, list[i]->z);
//	}
	#pragma omp parallel for schedule(dynamic)
	int i; 
	for (i = start; i < end; i++) {
		//DO hard work
		double x = list[i]->x;
		double y = list[i]->y;
		unsigned int num_shots = list[i]->num_shots;
#ifdef TEST
		printf("list[%d]: x = %lf, y = %lf\n",i, x, y);
		printf("list[%d]: walk_on_spheres(%lf, %lf, %u, %lf) = %lf\n",i , x, y, num_shots, EPSILON, WalkOnSpheresPolyTypeA(x, y, num_shots, EPSILON));
#endif
		list[i]->result = WalkOnSpheresPolyTypeA(x, y, num_shots, EPSILON);
#ifdef TEST
		printf("list[%d]->result = %lf\n",i, list[i]->result);
#endif
	}
	print_points(list+start, end-start, "");
	destroy_points(list, num_of_pts);
	time( &rawtime );
        timeinfo = localtime( &rawtime );
	//printf("Totally %d points, fails %d points.\n", end - start, fail_count);
#ifdef TEST
        printf( "Finish time and date: %s\n", asctime (timeinfo) );
#endif
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
	int i, j;
	int counter = 0;
	double x_d = 3.0/X_SLIDES;
	double y_d = 3.0/Y_SLIDES;
	int list_Size = NUM_POINTS;
	Point* list = malloc(sizeof(*list) * NUM_POINTS);
	if (!list) {
        	perror("Failed to allocate memory");
        	exit(EXIT_FAILURE);
    	}
	for (i = 0; i < list_Size; i++) list[i] = NULL;
	for (i = 0; i <= X_SLIDES; i++) {
		double x = i * x_d;
		for (j = 0; j <= Y_SLIDES; j++) {
			double y = j * y_d;
			//check validity
			if (x > 2 && y > 1) continue; //The point is not in the domain
			list[counter] = create_point(x, y);
			if (y > 1.5 || y < 0.5) list[counter]->num_shots = NUM_SHOTS *3;
			else list[counter]->num_shots = NUM_SHOTS;

			counter++;
		}
	}

	//Densify the numeric solution points at certain subregions
	for (i = 0; i <= 20; i++){
		double x = 1.9 + (i+1) * 0.01;
		for(j = 0; j <= 20; j++){
			double y = 0.95 + (j+1) * 0.0101;
			//check validity
			//if (x > 2 && y > 1) continue; //The point is not in the domain
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

#ifdef TEST
	printf("Generated %d points.\n", counter);
#endif
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
	int i;
	char name[15] = "points";
	strcat(name, str);
	strcat(name, ".txt");
	FILE* fd = fopen(name, "w");
	for (i = 0; i < num_of_pts; i++) {
		Point pt = list[i];
		fprintf(fd, "%lf, %lf, %lf\n", pt->x, pt->y, pt->result);
	}
	fclose(fd);
}

inline double find_min_radius(double x, double y) {
    if (y <= 1) {
        double dist1 = y;
        double dist2 = length(x - 2, y - 1);
        double dist3 = length(x - 4, y - 1);
        return (dist1 < dist2) ? ((dist1 < dist3) ? dist1 : dist3) : ((dist2 < dist3) ? dist2 : dist3);
    } else {
        if (x < 2) {
            return (2 - y < 2 - x) ? (2 - y) : (2 - x);
        } else if (x > 4) {
            return (2 - y < x - 4) ? (2 - y) : (x - 4);
        } else {
            return (x - 2 < 4 - x) ? (x - 2) : (4 - x);
        }
    }
//	printf("Find radius: %lf\n", l1>=l2?l2:l1);
}


inline void move_to_next(Point pt, double r) {
	double u1 = ((double)(rand()))*2*PI/(RAND_MAX);
	double z1 = sin(u1);
	double z2 = cos(u1);
	pt->x += r * z1;
	pt->y += r * z2;
}


inline void bounce_inside(Point pt) {
        if (pt->x < 0) pt->x = -(pt->x);  // Out of Neumann boundary x=0, 0<=y<=2, Reflection
	else if (pt->x > 6) pt->x = pt->x - 6;  // Out of Neumann boundary x=6, 0<=y<=2, Reflection and translation
        else if (2 < pt->x && pt->x < 4 && pt->y > 1) pt->y = 2 - pt->y;  // Out of Neumann boundary y=1, 2<=x<=4, Reflection
        // printf("Move to %.6lf, %.6lf, %.6lf\n", pt->x, pt->y, pt->z);
}


inline bool isOnDirichlet(Point pt, int* mark) {
	if ( 3-pt->y < EPSILON) {                    //(*If y < 0, it means the Dirichlet boundary is the left one*);
		(*mark)++;
		#ifndef TEST
	//	fprintf(stderr, "Right plane\n");
		#endif
		return true;
	}
	else if(pt->y < EPSILON){
		#ifndef TEST
	//	fprintf(stderr, "Up plane\n");
		#endif
		return true;
	}
	return false;
}

