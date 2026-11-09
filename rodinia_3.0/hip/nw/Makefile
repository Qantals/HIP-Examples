# Build the archived HIP Rodinia NW benchmark for gem5's VEGA_X86 model.
# The target list matches gem5-resources/src/gpu/square/Makefile.

HIP_PATH ?= /opt/rocm
HIPCC ?= $(HIP_PATH)/bin/hipcc
HIPCC_FLAGS ?= -O2
GEM5_AMDGPU_TARGETS ?= gfx900,gfx902

TARGET := needle
SOURCE := needle_hip.cu
HEADERS := needle.h needle_kernel_hip.cu

.PHONY: all clean

all: $(TARGET)

$(TARGET): $(SOURCE) $(HEADERS)
	$(HIPCC) $(HIPCC_FLAGS) \
		--amdgpu-target=$(GEM5_AMDGPU_TARGETS) $(SOURCE) -o $@ -lm

clean:
	rm -f $(TARGET) nw *.o *~ *.linkinfo *.perf result.txt
