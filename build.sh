#!/bin/sh

g++ -O3 -DNDEBUG -I lib -framework OpenGL -framework GLUT src/render.cpp -llapack -lblas  src/calcLayout.cpp lib/liblbfgs.dylib src/reprojection.cpp
