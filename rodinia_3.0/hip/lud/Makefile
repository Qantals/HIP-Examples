# Build the archived HIP Rodinia LUD benchmark for gem5 VEGA_X86.

HIP_PATH ?= /opt/rocm
HIPCC ?= $(HIP_PATH)/bin/hipcc
HIPCC_FLAGS ?= -O2
GEM5_AMDGPU_TARGETS ?= gfx900,gfx902

TARGET := lud_hip
HIP_OBJECTS := lud.hip.o lud_kernel.hip.o
COMMON_OBJECT := common/common.o

.PHONY: all clean

all: $(TARGET)

$(COMMON_OBJECT): common/common.c common/common.h
	$(CC) -O2 -Icommon -c $< -o $@

%.hip.o: %.hip.cu common/common.h
	$(HIPCC) $(HIPCC_FLAGS) --amdgpu-target=$(GEM5_AMDGPU_TARGETS) \
		-c $< -o $@

$(TARGET): $(HIP_OBJECTS) $(COMMON_OBJECT)
	$(HIPCC) $(HIPCC_FLAGS) -x none $(HIP_OBJECTS) $(COMMON_OBJECT) -o $@ -lm

clean:
	rm -f $(TARGET) $(HIP_OBJECTS) $(COMMON_OBJECT) *.linkinfo *.perf result.bin
