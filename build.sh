#!/bin/sh

g++ -O3 -I lib -framework OpenGL -framework GLUT src/render.cpp -framework vecLib src/calcLayout.cpp lib/liblbfgs.dylib src/reprojection.cpp
