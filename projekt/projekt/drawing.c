#include <GL/glew.h>
#include <GL/freeglut.h>
#include <GLFW/glfw3.h>
#include <stdio.h>
#include <math.h>


#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define SCREEN_WIDTH 1280
#define SCREEN_HEIGHT 720

//GLfloat gfPosX = 0.0;
//GLfloat gfDeltaX = 5.0/SCREEN_WIDTH;
//GLfloat gfPosY = 0.1;

int nPoints = 2;

GLfloat pos[2][3] = {
	{ 0.2, 0.2, 0.5 },
	{ 0.5, 0.5, 0.0 }
};

GLfloat vel[2][3] = {
	{ 0.001, -0.002, 0.0 },
	{ -0.003, 0.001, 0.0 }
};


GLfloat prog[2] = { 0.0, 0.0 };

void Draw(){
	glClear(GL_COLOR_BUFFER_BIT);
	glColor3f(1.0, 1.0, 1.0);
	
	glEnable(GL_POINT_SMOOTH);
	glPointSize(3);


	glBegin(GL_POINTS);
	for (int i = 0; i < nPoints; i++){
		glVertex3f(pos[i][0], pos[i][1], pos[i][2]);
		for (int j = 0; j < 3; j++)
		{
			pos[i][j] += vel[i][j];
		}
	}	
	glEnd();

	//glBegin(GL_POINTS);

	//GLfloat r1 = 0.2;
	//GLfloat r2 = 0.05;
	//GLfloat aspect = 1.0 * SCREEN_WIDTH / SCREEN_HEIGHT;
	//GLfloat planetPos[2] = {
	//	0.5 + r1 * cos(prog[0]),
	//	0.5 + r1 * aspect * sin(prog[0])
	//};
	//glVertex2d(planetPos[0], planetPos[1]);
	//glVertex2d(planetPos[0] + r2 * cos(prog[1]), planetPos[1] + r2 * aspect * sin(prog[1]));
	//prog[0] = fmod(prog[0] + 0.02, 2 * M_PI);
	//prog[1] = fmod(prog[1] + 0.07, 2 *  M_PI);

	//glEnd();

	glutSwapBuffers();
}

void Timer(int frameTimeMs){
	glutPostRedisplay();
	glutTimerFunc(frameTimeMs, Timer, 0);
}

void Initialize() {
	glClearColor(0.0, 0.0, 0.0, 0.0);
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	glOrtho(0.0, 1.0, 0.0, 1.0, -1.0, 1.0);
}


int main(int argc, char* argv[]) {
	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB);
	glutInitWindowSize(SCREEN_WIDTH, SCREEN_HEIGHT);
	glutInitWindowPosition(200, 200);
	glutCreateWindow("Galaxy");
	Initialize();
	glutDisplayFunc(Draw);

	Timer(14);
	glutMainLoop();
}
