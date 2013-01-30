#
# Makefile for autoconf tutorial
#

EXTRAINC = -I $(HOME)/Software/include
FRAMEWORK = -framework OpenGL
FRAMEWORK += -framework GLUT

CXX = g++
DEFS = -O3 -DNODEBUG $(EXTRAINC)
LBFGSLIB = -L $(HOME)/Software/lib -llbfgs
LIBS = $(FRAMEWORK) -llapack -lblas $(LBFGSLIB)

SRCS = render.cpp calcLayout.cpp reprojection.cpp
OBJS = $(SRCS:.cpp=.o)
PROG = agi3d

all: $(PROG)

$(PROG): $(OBJS)
	$(CXX) -o $@ $(OBJS) $(LIBS)

.cpp.o:
	$(CXX) $(DEFS) -c -o $@ $<

clean:
	rm -f $(OBJS)

veryclean: clean
	rm $(PROG)
