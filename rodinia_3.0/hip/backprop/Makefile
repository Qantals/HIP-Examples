# Build archived HIP Rodinia backprop for gem5 VEGA_X86.

HIP_PATH ?= /opt/rocm
HIPCC ?= $(HIP_PATH)/bin/hipcc
HIPCC_FLAGS ?= -O2
GEM5_AMDGPU_TARGETS ?= gfx900,gfx902

TARGET := backprop
HOST_OBJECTS := backprop.o facetrain.o imagenet.o
HIP_OBJECT := backprop_cuda.o

.PHONY: all clean

all: $(TARGET)

$(HOST_OBJECTS): %.o: %.c backprop.h
	$(CC) -O2 -c $< -o $@

$(HIP_OBJECT): backprop_cuda.cu backprop_cuda_kernel.cu backprop.h
	$(HIPCC) $(HIPCC_FLAGS) --amdgpu-target=$(GEM5_AMDGPU_TARGETS) \
		-c $< -o $@

$(TARGET): $(HOST_OBJECTS) $(HIP_OBJECT)
	$(HIPCC) $(HIPCC_FLAGS) -x none $^ -o $@ -lm

clean:
	rm -f $(TARGET) $(HOST_OBJECTS) $(HIP_OBJECT) *.linkinfo *.perf
