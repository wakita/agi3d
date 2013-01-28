#include <GL/glut.h>
#include <iostream>
#include <string>
#include <new>
#include <cml/cml.h>
#include "stopwatch.h"
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/vector.hpp>
typedef boost::numeric::ublas::matrix<float> fmatrix;
typedef boost::numeric::ublas::vector<float> fvector; 
typedef cml::vector3f vector3;
typedef cml::vector4f vector4;
typedef cml::matrix44f_c matrix;
using namespace std;

//Params about Graph
extern int N;
extern int M;
//vector<int> *neighbor;

//Graph Layouts
extern fmatrix C;
extern vector< pair<int, int> >  edges;
void loadData(char *);
void calcInitLayout();

static float * pos_x, * pos_y, * pos_z;
static vector3 * colors;
static vector3 blue(0.0f, 0.0f, 1.0f);
static vector3 red(1.0f,0.0f,0.0f);

//window
static int width = 800; static int height = 800;

//camera
static float v = 6.0f;
static vector3 eye(0,0,v);
static vector3 target(0,0,0);
static vector3 up(0,1,0);
static float phi = 0, theta = 0;
static float radius = 0.05f;

//light
static vector4 lightPos(eye[0],eye[1],eye[2],1.0f);

//Perspective
static float angle = 45.0f, near = 1.0f, far = 60.0f;

//Mouse Adaption
static bool isPicked = false, isDrag = false;
static int mouse_pos_x = 0; static int mouse_pos_y = 0;
static int id = -1;
static float pre_x, pre_y, pre_z;
int reprojection(int, float, float, float, float, float, float);

//Buffer
static GLuint points;
static GLuint buffer[2];
typedef GLfloat Position[3];
typedef GLuint Face[3];
static double _matrix[16];

GLuint solidSphere(float radius, int slices, int stacks, const GLuint *buffer){
    GLuint vertices = (slices + 1) * (stacks + 1);
    GLuint faces = slices * stacks * 2;
    
    glBindBuffer(GL_ARRAY_BUFFER, buffer[0]);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, buffer[1]);
    glBufferData(GL_ARRAY_BUFFER, sizeof (Position) * vertices, NULL, GL_STATIC_DRAW);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof (Face) * faces, NULL, GL_STATIC_DRAW);
    
    Position *position = (Position *)glMapBuffer(GL_ARRAY_BUFFER, GL_WRITE_ONLY);
    Face *face = (Face *)glMapBuffer(GL_ELEMENT_ARRAY_BUFFER, GL_WRITE_ONLY);
    
    for (int j = 0; j <= stacks; ++j) {
        float ph = 3.141593f * (float)j / (float)stacks;
        float y = radius*cosf(ph);
        float r = radius*sinf(ph);
        for (int i = 0; i <= slices; ++i) {
            float th = 2.0f * 3.141593f * (float)i / (float)slices;
            float x = r * cosf(th), z = r * sinf(th);
            (*position)[0] = x; (*position)[1] = y; (*position)[2] = z;
            ++position;
        }
    }

    for (int j = 0; j < stacks; ++j) {
        for (int i = 0; i < slices; ++i) {
            int count = (slices + 1) * j + i;
            (*face)[0] = count; (*face)[1] = count + 1; (*face)[2] = count + slices + 2;
            ++face;
            (*face)[0] = count; (*face)[1] = count + slices + 2; (*face)[2] = count + slices + 1;
            ++face;
        }
    }

    glUnmapBuffer(GL_ELEMENT_ARRAY_BUFFER);
    glUnmapBuffer(GL_ARRAY_BUFFER);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    return faces * 3;
}

void Lighting(){
    glEnable(GL_COLOR_MATERIAL);
    glColorMaterial(GL_FRONT, GL_AMBIENT_AND_DIFFUSE);
    glEnable(GL_LIGHTING);
    GLfloat light_ambient[] = {0.2f,0.2f,0.2f,1.0f};
    GLfloat light_diffuse[] = {0.65f,0.65f,0.65f,1.0f};
    GLfloat light_position[] = {0.0f,0.0f,-v,1.0f};
    glLightfv(GL_LIGHT0, GL_AMBIENT,  light_ambient);
    glLightfv(GL_LIGHT0, GL_DIFFUSE,  light_diffuse);
    glLightfv(GL_LIGHT0, GL_POSITION, lightPos.data());
    glLightfv(GL_LIGHT1, GL_AMBIENT,  light_ambient);
    glLightfv(GL_LIGHT1, GL_DIFFUSE,  light_diffuse);
    glLightfv(GL_LIGHT1, GL_POSITION, light_position);
    //	glLightfv(GL_LIGHT0, GL_POSITION, light0_position);
    glEnable(GL_LIGHT0);
    glEnable(GL_LIGHT1);
}

void RenderScene(){
    static int iFrames = 0;
    static CStopWatch frameTimer;   
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glEnable(GL_DEPTH_TEST);

    //ModelView
    glMatrixMode(GL_MODELVIEW);
    glPushMatrix();

    Lighting();

    //Camera Position
    matrix xRot, yRot, view;
    cml::matrix_rotation_world_x(xRot,-theta);
    cml::matrix_rotation_world_y(yRot,phi);
    up.set(0.0f,1.0f,0.0f);
    vector4 _up(up[0],up[1],up[2],1);
    _up = yRot*xRot*_up;
    up.set(_up[0],_up[1],_up[2]);
    up.normalize();
    cml::matrix_look_at_RH(view, eye, target, up);
    glLoadMatrixf(view.data());

    // Initialize the names stack
    glInitNames();
    glPushName(N);

    glEnableVertexAttribArray(0);
    glBindBuffer(GL_ARRAY_BUFFER, buffer[0]);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, 0);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, buffer[1]);

    //Draw Nodes
    for (int i = 0; i < N; i++) {
        glPushMatrix();
        glColor3fv(colors[i].data());
        glTranslatef((GLfloat)pos_x[i], (GLfloat)pos_y[i], (GLfloat)pos_z[i]);
        glLoadName(i);
        glDrawElements(GL_TRIANGLES, points, GL_UNSIGNED_INT, 0);
        glPopMatrix();
    }

    //Draw Edges
    glColor3f(0.15f,0.15f,0.15f);
    for(int i = 0; i < M; i ++){
        int from = edges[i].first, to = edges[i].second;
        glBegin(GL_LINES);
        glVertex3f(pos_x[from],pos_y[from],pos_z[from]);
        glVertex3f(pos_x[to],pos_y[to],pos_z[to]);
        glEnd();
    }

    glPopMatrix();

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glDisableVertexAttribArray(0);
    glDisable(GL_DEPTH_TEST);

    Lighting();
    glutSwapBuffers();

    iFrames++;
    if(iFrames == 100){
        float fps; char cBuffer[64];
        fps = 100.0f / frameTimer.GetElapsedSeconds();
        sprintf(cBuffer,"FPS : %.1f fps", fps);
        glutSetWindowTitle(cBuffer);
        frameTimer.Reset();
        iFrames = 0;
    }
}

#define BUFFER_LENGTH 64
void ProcessSelection(int xPos, int yPos){
    GLfloat fAspect;
    static GLuint selectBuff[BUFFER_LENGTH];
    GLint hits, viewport[4];

    glSelectBuffer(BUFFER_LENGTH, selectBuff);
    glGetIntegerv(GL_VIEWPORT, viewport);

    glMatrixMode(GL_PROJECTION);
    glPushMatrix();

    glRenderMode(GL_SELECT);
    glLoadIdentity();
    gluPickMatrix(xPos, viewport[3] - yPos + viewport[1], 1,1, viewport);
    fAspect = (float)viewport[2] / (float)viewport[3];
    gluPerspective(angle, fAspect, near, far);
    RenderScene();
    hits = glRenderMode(GL_RENDER);
    
    if(hits >= 1 && !isPicked){
        //cout << "PICK" << endl;
        for(int i = 3; i < BUFFER_LENGTH; i+= 3){
            int tmp = selectBuff[i];
            cout << tmp << endl;
            if(tmp >= 0 && tmp < N){
                id = tmp;
                break;
            }
        }
        cout << id << endl;
        pre_x = pos_x[id];
        pre_y = pos_y[id];
        pre_z = pos_z[id]; 
        colors[id] = red;
        isPicked = true;
    }
    
    glMatrixMode(GL_PROJECTION);
    glPopMatrix();
    glMatrixMode(GL_MODELVIEW);
}

void scene(){}

void relayout(){
    for(int i = 0; i < N; i++){
        pos_x[i] = C(i,0);
        pos_y[i] = C(i,1);
        pos_z[i] = C(i,2);
    }
}

void MouseCallback(int button, int state, int x, int y){
    if(button == GLUT_LEFT_BUTTON && state == GLUT_DOWN){
        if(!isPicked)
            ProcessSelection(x, y);
        if(!isDrag)
            isDrag = true;
        mouse_pos_x = x;
        mouse_pos_y = y;
    }
    if(button == GLUT_LEFT_BUTTON && state == GLUT_UP){
        isPicked = false;
        isDrag = false;
        colors[id] = blue;
        id = -1;
        glutPostRedisplay();
    }
}

void MouseMotionCallback(int x, int y){
    if(isDrag && !isPicked){
        theta -= 0.005*(y - mouse_pos_y);
        phi +=  0.005*(x - mouse_pos_x);
        
        float camera_y = v * (float)sin(theta);
        float camera_xz = v * (float)cos(theta);
        float camera_x =  camera_xz * (float)sin(phi);
        float camera_z =  camera_xz * (float)cos(phi);
        
        eye.set(camera_x, camera_y, camera_z);
        lightPos.set(eye[0],eye[1],eye[2],1.0f);
        
        mouse_pos_x = x; mouse_pos_y = y;
        
        glutPostRedisplay();
    }
    else if(isDrag && isPicked){
        int dx = x - mouse_pos_x;
        int dy = y - mouse_pos_y;
        
        vector3 pos(pos_x[id],pos_y[id],pos_z[id]);
        vector3 v = target - eye;
        vector3 right = cml::cross(up,v);
        right.normalize();
        float dist = cml::length(pos-eye);
        
        //Proper Rate is ???
        float rate_x = (float)(-dx*dist)/(float)width;
        float rate_y =  (float)(-dy*dist)/(float)height;
        
        vector3 delta = rate_x * right + rate_y * up;
        
        float new_x = pos_x[id] + delta[0];
        float new_y = pos_y[id] + delta[1];
        float new_z = pos_z[id] + delta[2];
        
        mouse_pos_x = x; mouse_pos_y = y;
        
        int a = reprojection(id, pre_x, pre_y, pre_z, new_x, new_y, new_z);
        
        if(a == 1){
            relayout();
            pre_x = pos_x[id];
            pre_y = pos_y[id];
            pre_z = pos_z[id];
            pos_x[id] = new_x;
            pos_y[id] = new_y;
            pos_z[id] = new_z;
        }
        
        glutPostRedisplay();
    }
}

void idle(){
    //relayout();
}

void SetupRC(){
    pos_x = new float[N]; 
    pos_y = new float[N];
    pos_z = new float[N];
    colors = new vector3[N];
    
    for(int i = 0; i < N; i++){
        pos_x[i] = C(i,0);
        pos_y[i] = C(i,1);
        pos_z[i] = C(i,2);
        colors[i] = blue;
    }
    
    //camera setting
    eye.set(0,0,v); target.zero(); up.set(0.0f,1.0f,0.0f);
    glEnable(GL_DEPTH_TEST);
    
    glEnable(GL_NORMALIZE);

    //culling
    glFrontFace(GL_CCW);
    glEnable(GL_CULL_FACE);
    glCullFace(GL_FRONT);
    //Lighting();
    //glEnable(GL_LINE_SMOOTH);
    glLineWidth(0.01f);
    //Back Ground
    glClearColor(0.9f, 0.9f, 0.9f, 1.0f );
    //Buffer
    glGenBuffers(2, buffer);
    points = solidSphere(radius, 10, 5, buffer);
}

void ChangeSize(int w, int h){
    GLfloat fAspect;
    if(h == 0) h = 1;
    
    glViewport(0, 0, w, h);
    fAspect = (GLfloat)w/(GLfloat)h;
    
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluPerspective(angle, fAspect, near, far);
    
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    width = w; height = h;
}

int main(int argc, char* argv[]){
    loadData(argv[1]);
    pos_x = new float[N];
    pos_y = new float[N];
    pos_z = new float[N];
    colors = new vector3[N];
    
    for(int i = 0; i < N; i++){
        colors[i] = blue;
    }

    calcInitLayout();

    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH);
    glutInitWindowSize(width,height);
    
    glutCreateWindow("window");
    glutReshapeFunc(ChangeSize);
    glutMouseFunc(MouseCallback);
    glutMotionFunc(MouseMotionCallback);
    glutDisplayFunc(RenderScene);
    glutIdleFunc(idle);

    SetupRC();
    glutMainLoop();

    return 0;
}
