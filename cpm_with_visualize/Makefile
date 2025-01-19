# Compiler
CXX = g++
CC = gcc
CUDA_PATH = /usr/local/cuda
NVCC = $(CUDA_PATH)/bin/nvcc
#CUDA_FLAGS = -O3 -arch=sm_50
CUDA_FLAGS = -O0 -g -arch=sm_50 -DDEBUG
CUDA_LIBS = -lcuda -lcudart -lcusparse -lcusolver
FLAGS =  -DDEBUG

# Programs
PROG1 = cpm
PROG2 = visualize
PROG3 = cpm_cuda
PROG_MAIN = main

# Source files
SRCS1 = cpm.c cpm_functions.c calc_time.c
SRCS2 = visualize.cpp
SRCS3 = cpm_cuda.cu calc_time.c cpm_functions.c cuda_functions.cu
#SRCS3 = cpm.c calc_time.c cpm_functions.c cpm_cuda_functions.cu
SRCS_MAIN = main.cpp

# Object files
OBJS1 = cpm.o calc_time.o cpm_functions.o
OBJS2 = $(SRCS2:.cpp=.o)
OBJS3 = cpm.o calc_time.o cpm_functions.o cpm_cuda_functions.o
OBJS_MAIN = $(SRCS_MAIN:.cpp=.o)

# sparse_solver
SPARSE_SOLVER_OBJS = sparse_solver.o
SPARSE_SOLVER_LIB = libsparsesolver.a


# Executables
EXEC1 = cpm.out
EXEC2 = visualize.out
EXEC3 = cpm_cuda.out
EXEC_MAIN = main.out

# Variabile condizionale per includere librerie e percorsi specifici
ifeq ($(LOGNAME),annagreco)
	INCLUDEFLAGS = -I/opt/homebrew/Cellar/lapack/3.12.0/include -I/opt/homebrew/Cellar/suite-sparse/7.8.2/include/suitesparse
	LIBRARYFLAGS = -L/opt/homebrew/Cellar/lapack/3.12.0/lib -L/opt/homebrew/Cellar/suite-sparse/7.8.2/lib
	OPENCV_FLAGS = pkg-config --cflags --libs opencv4
	QT_FLAGS = pkg-config --cflags --libs Qt5Widgets
else ifeq ($(LOGNAME),greco)
	INCLUDEFLAGS = -I/home/greco/miniconda3/envs/sparsematrix/include/suitesparse -I/usr/local/cuda/include
	LIBRARYFLAGS = -L/home/greco/miniconda3/envs/sparsematrix/lib -L/usr/local/cuda/lib64 -L/home/greco/miniconda3/lib
	OPENCV_FLAGS = $(shell pkg-config --cflags --libs opencv4)
	QT_FLAGS = $(shell pkg-config --cflags --libs Qt5Widgets)
else ifeq ($(LOGNAME),anna)
	INCLUDEFLAGS = -I/usr/include -I/usr/include/suitesparse
	LIBRARYFLAGS = -L/usr/lib -L/usr/lib/x86_64-linux-gnu/libsuitesparse
	OPENCV_FLAGS = $(shell pkg-config --cflags --libs opencv4)
	QT_FLAGS = $(shell pkg-config --cflags --libs Qt5Widgets)
	CUDA_INCLUDE = -I/usr/local/cuda/include
else
	INCLUDEFLAGS =
	LIBRARYFLAGS =
	OPENCV_FLAGS = pkg-config --cflags --libs opencv4
	QT_FLAGS = pkg-config --cflags --libs Qt5Widgets
endif

# Default flags
CFLAGS = -Wall -Wextra $(INCLUDEFLAGS) -fPIE $(FLAGS)
CXXFLAGS = -Wall -Wextra $(INCLUDEFLAGS) $(OPENCV_FLAGS) $(QT_FLAGS) -std=c++17 -fPIE $(FLAGS)
CPPFLAGS = -MMD -MP $(FLAGS)

# Debug flags
DEBUG_CFLAGS = -g -O0 -static
DEBUG_CXXFLAGS = -g -O0 -static

# Release flags
RELEASE_CFLAGS = -O0 -static -DDEBUG

# LAPACK library flags
LDFLAGS =
LDFLAGS1 = $(LIBRARYFLAGS) -llapacke -llapack -lblas -lm
LDFLAGS2 = $(LIBRARYFLAGS) \
	-lcxsparse -lsuitesparseconfig -lumfpack -lamd -lcholmod -lcolamd -lcamd -lccolamd \
	-lavcodec -lavformat -lavutil -lopus -lopusfile -lvorbis -lrsvg-2 -lgdk_pixbuf-2.0 -lcairo -ltiff -lLerc -lgio-2.0 -lresolv -lwebp -lzstd -lrt -llzma -ljpeg -ldeflate -lgmodule-2.0 -ldl -lgobject-2.0 -lffi -lglib-2.0 -latomic -lpcre2-8 -lfontconfig -lXext -lXrender -lxcb-render -lX11-xcb -lxcb-shm -lpixman-1 -lexpat -lfreetype -lpng16 -lz -lX11 -lxcb -lXau -lXdmcp \
	-lharfbuzz -lpango-1.0 -lpangocairo-1.0 \
	-lQt5Core -lQt5Network -lQt5DBus -lQt5Gui -lQt5Widgets -lQt5Test \
	-lpthread -lvpx -ldav1d -laom -lmp3lame -lx264 -lx265 -lopenh264 -lswresample -lva-drm -lva-x11 -ldrm -lssl -lcrypto -lva \
	-lstdc++ -lm

# Default target
all: $(EXEC1) $(EXEC2) $(EXEC3) $(EXEC_MAIN)

# Debug targets
debug: debug_cpm

debug_cpm: CFLAGS += $(DEBUG_CFLAGS)
debug_cpm: $(EXEC1)
	@echo "Debug build of $(EXEC1) complete"

# Release targets
release: release_cpm

release_cpm: CFLAGS += $(RELEASE_CFLAGS)
release_cpm: $(EXEC1)
	@echo "Release build of $(EXEC1) complete"

# Linking for C program
$(EXEC1): $(OBJS1)
	$(CC) $(CFLAGS) $(OBJS1) -o $(EXEC1) $(LDFLAGS1) $(LDFLAGS2) -Wl,-rpath $(LIBRARYFLAGS)


# Linking for C++ program
$(EXEC2): $(OBJS2)
	$(CXX) $(CXXFLAGS) $(OBJS2) -o $(EXEC2) $(LDFLAGS2)  $(OPENCV_FLAGS)

# Linking for main program
$(EXEC_MAIN): $(OBJS_MAIN)
	$(CXX) $(CXXFLAGS) $(OBJS_MAIN) -o $(EXEC_MAIN) $(LDFLAGS2)

# Compilation of C source files
cpm.o: cpm.c cpm.h write_matrix.h calc_time.h
	$(CC) $(CPPFLAGS) $(CFLAGS) -c $< -o $@

calc_time.o: calc_time.c calc_time.h
	$(CC) $(CPPFLAGS) $(CFLAGS) -c $< -o $@

cpm_functions.o: cpm_functions.c cpm.h
	$(CC) $(CPPFLAGS) $(CFLAGS) -c $< -o $@

main.o: main.c
	$(CC) $(CPPFLAGS) $(CFLAGS) -c $< -o $@

# Compilation of C++ source files
visualize.o: visualize.cpp visualize.hpp
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -c $< -o $@


# Compilation of CUDA source files
cpm_cuda_functions.o: cpm_cuda_functions.cu cpm_cuda_functions.cuh sparse_solver.cuh
	$(NVCC) $(CUDA_FLAGS) $(INCLUDEFLAGS) $(CUDA_INCLUDE) -g -c $< -o $@

cuda_functions.o: cuda_functions.cu cuda_functions.cuh write_matrix.h calc_time.h cpm.h
	$(NVCC) $(CUDA_FLAGS) $(INCLUDEFLAGS) $(CUDA_INCLUDE) -g -c $< -o $@

#	$(NVCC) $(INCLUDEFLAGS) -rdc=true -Xcompiler "-fPIC -fPIE" -c $< -o $@

cpm_cuda.o: cpm_cuda.cu cpm_cuda.cuh write_matrix.h calc_time.h
	$(NVCC) $(CUDA_FLAGS) $(INCLUDEFLAGS) $(CUDA_INCLUDE) -g -c $< -o $@

#   $(NVCC) $(INCLUDEFLAGS) -rdc=true -Xcompiler "-fPIC -fPIE" -c $< -o $@

sparse_solver.o: sparse_solver.cu sparse_solver.cuh
	$(NVCC) $(CUDA_FLAGS) $(CUDA_INCLUDE) -g -c $< -o $@

#	$(NVCC) $(INCLUDEFLAGS) -rdc=true -Xcompiler "-fPIC -fPIE" -c $< -o $@

# Regola per creare la libreria statica
$(SPARSE_SOLVER_LIB): $(SPARSE_SOLVER_OBJS)
	ar rcs $@ $^

cpm_cuda.out: cpm_cuda.o sparse_solver.o calc_time.o cpm_cuda_functions.o cuda_functions.o $(SPARSE_SOLVER_LIB)
	$(NVCC) $(CUDA_FLAGS) $(CUDA_INCLUDE) $^  $(CUDA_LIBS) $(LDFLAGS1) $(LDFLAGS2) -o $@

# Include dependency files
-include $(OBJS1:.o=.d) $(OBJS2:.o=.d) $(OBJS3:.o=.d) $(OBJS_MAIN:.o=.d)

# Clean up
clean:
	rm -f $(OBJS1) $(OBJS2) $(OBJS3) $(OBJS_MAIN) $(EXEC1) $(EXEC2) $(EXEC3) $(EXEC_MAIN) $(OBJS1:.o=.d) $(OBJS2:.o=.d) $(OBJS3:.o=.d) $(OBJS_MAIN:.o=.d) cuda*.o