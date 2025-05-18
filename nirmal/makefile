# Compiler settings
MPICXX = mpicxx
NVCC = nvcc

# Flags
CXXFLAGS = -std=c++14 -O3 -fopenmp
NVCCFLAGS = -std=c++14 -O3 -Xcompiler -fopenmp
LDFLAGS = -lcudart

# Source files
CPP_SOURCES = main.cpp matrix.cpp
CU_SOURCES = gpu_matrix.cu

# Object files
CPP_OBJECTS = $(CPP_SOURCES:.cpp=.o)
CU_OBJECTS = $(CU_SOURCES:.cu=.o)

# Headers
HEADERS = matrix.h gpu_matrix.h

# Target
TARGET = a4

all: $(TARGET)

$(TARGET): $(CPP_OBJECTS) $(CU_OBJECTS)
	$(MPICXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)

%.o: %.cpp $(HEADERS)
	$(MPICXX) $(CXXFLAGS) -c $< -o $@

%.o: %.cu $(HEADERS)
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

clean:
	rm -f $(CPP_OBJECTS) $(CU_OBJECTS) $(TARGET)