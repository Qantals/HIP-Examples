# Build the archived HIP Rodinia Pathfinder benchmark for gem5 VEGA_X86.

HIP_PATH ?= /opt/rocm
HIPCC ?= $(HIP_PATH)/bin/hipcc
HIPCC_FLAGS ?= -O2
GEM5_AMDGPU_TARGETS ?= gfx900,gfx902

.PHONY: all clean

all: pathfinder

pathfinder: pathfinder_hip.cu
	$(HIPCC) $(HIPCC_FLAGS) --amdgpu-target=$(GEM5_AMDGPU_TARGETS) $< -o $@ -lm

clean:
	rm -f pathfinder *.o *.linkinfo *.perf result.txt
