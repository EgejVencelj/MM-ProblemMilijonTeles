#include <GL/glew.h>
#include <GL/freeglut.h>
#include <GLFW/glfw3.h>
#include <stdio.h>

#define SCREEN_WIDTH 800
#define SCREEN_HEIGHT 600

GLfloat gfPosX = 0.0;
GLfloat gfDeltaX = .01;

void Draw(){
	glClear(GL_COLOR_BUFFER_BIT);
	glColor3f(1.0, 1.0, 1.0);
	glPointSize(20);
	glBegin(GL_POINTS);
		glVertex3f(gfPosX, 0.25, 0.0);
	glEnd();
	glutSwapBuffers();
	gfPosX += gfDeltaX;
	if (gfPosX >= 1.0 || gfPosX <= 0.0){
		gfDeltaX = -gfDeltaX;
	}
}

void Timer(int unused){
	glutPostRedisplay();
	glutTimerFunc(30, Timer, 0);
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

	Timer(0);
	glutMainLoop();
}
