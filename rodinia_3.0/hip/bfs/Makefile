# Build the archived HIP Rodinia BFS benchmark for gem5's VEGA_X86 model.
# The target list matches gem5-resources/src/gpu/square/Makefile.

HIP_PATH ?= /opt/rocm
HIPCC ?= $(HIP_PATH)/bin/hipcc
HIPCC_FLAGS ?= -O2
GEM5_AMDGPU_TARGETS ?= gfx900,gfx902

TARGET := bfs
SOURCE := bfs.hip.cu

.PHONY: all clean

all: $(TARGET)

$(TARGET): $(SOURCE) kernel.hip.cu kernel2.hip.cu
	$(HIPCC) $(HIPCC_FLAGS) \
		--amdgpu-target=$(GEM5_AMDGPU_TARGETS) $(SOURCE) -o $@ -lm

clean:
	rm -f $(TARGET) *.o result.txt *.perf
