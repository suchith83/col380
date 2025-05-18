# ========================
# Makefile for final2.cu
# ========================

NVCC      := nvcc
CXXFLAGS  := -std=c++14 -Xcompiler "-fopenmp -O3"
CCBIN     := mpicxx
INCLUDES  :=
LIBS      := -lcudart

TARGET    := a4       
SRC       := final2.cu

all: $(TARGET)

$(TARGET): $(SRC)
	$(NVCC) $(CXXFLAGS) -ccbin $(CCBIN) $(INCLUDES) $< $(LIBS) -o $@

clean:
	rm -f $(TARGET)
