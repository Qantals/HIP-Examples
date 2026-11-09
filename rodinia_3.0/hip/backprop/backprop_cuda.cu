#include "hip/hip_runtime.h"


// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#ifdef __CUDA_CC__
#include "cuda.h"
#endif
#include <sys/time.h>

// includes, kernels
#include "backprop_cuda_kernel.cu"
#include "backprop.h"

#ifdef PROFILING
#include "RDTimer.h"
#endif

#define HIP_CHECK(call) do { \
  hipError_t error = (call); \
  if (error != hipSuccess) { \
    fprintf(stderr, "HIP error at %s:%d: %s\n", __FILE__, __LINE__, hipGetErrorString(error)); \
    exit(EXIT_FAILURE); \
  } \
} while (0)

////////////////////////////////////////////////////////////////////////////////

extern "C"
void bpnn_layerforward(float *l1, float *l2, float **conn, int n1, int n2);

extern "C"
void bpnn_output_error(float *delta, float *target, float *output, int nj, float *err);

extern "C"
void bpnn_hidden_error(float *delta_h, int nh, float *delta_o, int no, float **who, float *hidden, float *err);

extern "C" 
void bpnn_adjust_weights(float *delta, int ndelta, float *ly, int nly, float **w, float **oldw);


extern "C"
int setup(int argc, char** argv);

extern "C"
float **alloc_2d_dbl(int m, int n);

extern "C"
float squash(float x);

double gettime() {
  struct timeval t;
  gettimeofday(&t,NULL);
  return t.tv_sec+t.tv_usec*1e-6;
}

unsigned int num_threads = 0;
unsigned int num_blocks = 0;

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int
main( int argc, char** argv) 
{
	setup(argc, argv);
}


extern "C"
void bpnn_train_cuda(BPNN *net, float *eo, float *eh)
{
  int in, hid, out;
  float out_err, hid_err;
  
  in = net->input_n;
  hid = net->hidden_n;
  out = net->output_n;   
   
#ifdef GPU  
  int m = 0;
  float *input_hidden_cuda;
  float *input_cuda;
  float *output_hidden_cuda;
  float *partial_sum;
  float *hidden_partial_sum;
  float *hidden_delta_cuda;
  float *input_prev_weights_cuda;
  float sum;
  float *input_weights_one_dim;
  float *input_weights_prev_one_dim;
  float raw_sum[WIDTH + 1];
  num_blocks = in / 16;  
  dim3  grid( 1 , num_blocks);
  dim3  threads(16 , 16);

/* overall time - start */
#ifdef PROFILING
	float alloc_t1, cpu_to_gpu_t1,kernel_t1,gpu_to_cpu_t1,alloc_t2, cpu_to_gpu_t2,kernel_t2,gpu_to_cpu_t2,overall_cpu_t;
        RDTimerCPU* rdtimerOverallCpu = new RDTimerCPU();
        rdtimerOverallCpu->Reset("Overall CPU Time");
        rdtimerOverallCpu->Start();
#endif
  
  input_weights_one_dim = (float *) malloc((in + 1)* (hid + 1) * sizeof(float));
  input_weights_prev_one_dim = (float *) malloc((in + 1)* (hid + 1) * sizeof(float));
  partial_sum = (float *) malloc(num_blocks * WIDTH * sizeof(float));
  if (!input_weights_one_dim || !input_weights_prev_one_dim || !partial_sum) {
    fprintf(stderr, "Could not allocate backprop host buffers\n");
    exit(EXIT_FAILURE);
  }
 
  // this preprocessing stage is added to correct the bugs of wrong memcopy using two-dimensional net->inputweights
  for (int k = 0; k <= in; k++) {	
   for (int j = 0; j <= hid; j++) {
	  input_weights_one_dim[m] = net->input_weights[k][j];
	  input_weights_prev_one_dim[m] = net-> input_prev_weights[k][j];
	  m++;
    }
  }

/* malloc time-start */ 
#ifdef PROFILING
        SimplePerfSerializer* serializeTime = new SimplePerfSerializer("./backprop" );

        RDTimerCPU* rdtimercpu = new RDTimerCPU();

        rdtimercpu->Reset("Malloc Time");
        rdtimercpu->Start();
#endif
  
  HIP_CHECK(hipMalloc((void**) &input_cuda, (in + 1) * sizeof(float)));
  HIP_CHECK(hipMalloc((void**) &output_hidden_cuda, (hid + 1) * sizeof(float)));
  HIP_CHECK(hipMalloc((void**) &input_hidden_cuda, (in + 1) * (hid + 1) * sizeof(float)));
  HIP_CHECK(hipMalloc((void**) &hidden_partial_sum, num_blocks * WIDTH * sizeof(float)));
/* malloc time-stop, cpu-gpu transfer start */ 
#ifdef PROFILING
        alloc_t1 = rdtimercpu->Stop();
        serializeTime->Serialize(rdtimercpu);
#endif  
  
#endif

#ifdef CPU

  printf("Performing CPU computation\n");
  bpnn_layerforward(net->input_units, net->hidden_units,net->input_weights, in, hid);

#endif

#ifdef GPU
 
  printf("Performing GPU computation\n");
  
  //printf("in= %d, hid = %d, numblocks = %d\n", in, hid, num_blocks);

/* malloc time-stop, cpu-gpu transfer start */ 
#ifdef PROFILING
        rdtimercpu->Reset("CPU to GPU Transfer Time");
        rdtimercpu->Start();
#endif
  HIP_CHECK(hipMemcpy(input_cuda, net->input_units, (in + 1) * sizeof(float), hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(input_hidden_cuda, input_weights_one_dim, (in + 1) * (hid + 1) * sizeof(float), hipMemcpyHostToDevice));

/*cpu-gpu transfer-stop, kernel exec-start */
#ifdef PROFILING
    cpu_to_gpu_t1 =  rdtimercpu->Stop();
    serializeTime->Serialize(rdtimercpu);
    rdtimercpu->Reset("COMPUTE:Kernel Execution Time");
    //hipDeviceSynchronize();
    rdtimercpu->Start();
#endif
  hipLaunchKernelGGL(bpnn_layerforward_CUDA, dim3(grid), dim3(threads ), 0, 0, input_cuda,
	                                          output_hidden_cuda,
											  input_hidden_cuda,
											  hidden_partial_sum,
											  in,
											  hid);
 /* kernel exec-stop, gpu-cpu transfer-start */
#ifdef PROFILING
    kernel_t1 = rdtimercpu->Stop();
    serializeTime->Serialize(rdtimercpu);
#endif
  HIP_CHECK(hipDeviceSynchronize());
  
  hipError_t error = hipGetLastError();
	if (error != hipSuccess) {
		printf("bpnn kernel error: %s\n", hipGetErrorString(error));
		exit(EXIT_FAILURE);
	}

/* kernel exec-stop, gpu-cpu transfer-start */
#ifdef PROFILING
    rdtimercpu->Reset("GPU to CPU Transfer Time");
    rdtimercpu->Start();
#endif  
  HIP_CHECK(hipMemcpy(partial_sum, hidden_partial_sum, num_blocks * WIDTH * sizeof(float), hipMemcpyDeviceToHost));
/* gpu-cpu transfer- stop */ 
#ifdef PROFILING        
    gpu_to_cpu_t1= rdtimercpu->Stop();
    serializeTime->Serialize(rdtimercpu);
#endif     
  for (int j = 1; j <= hid; j++) {
    sum = 0.0;
    for (int k = 0; k < num_blocks; k++) {	
      sum += partial_sum[k * hid + j-1] ;
    }
    raw_sum[j] = sum;
    sum += net->input_weights[0][j];
	net-> hidden_units[j] = float(1.0 / (1.0 + exp(-sum)));
  }
  #endif

  bpnn_layerforward(net->hidden_units, net->output_units, net->hidden_weights, hid, out);
  bpnn_output_error(net->output_delta, net->target, net->output_units, out, &out_err);
  bpnn_hidden_error(net->hidden_delta, hid, net->output_delta, out, net->hidden_weights, net->hidden_units, &hid_err);  
  bpnn_adjust_weights(net->output_delta, out, net->hidden_units, hid, net->hidden_weights, net->hidden_prev_weights);

#ifdef CPU

  bpnn_adjust_weights(net->hidden_delta, hid, net->input_units, in, net->input_weights, net->input_prev_weights);

#endif  


#ifdef GPU

/* malloc time-start */
#ifdef PROFILING
        rdtimercpu->Reset("Malloc Time 2");
        rdtimercpu->Start();
#endif

  HIP_CHECK(hipMalloc((void**) &hidden_delta_cuda, (hid + 1) * sizeof(float)));
  HIP_CHECK(hipMalloc((void**) &input_prev_weights_cuda, (in + 1) * (hid + 1) * sizeof(float)));

/* malloc time-stop, cpu-gpu transfer start */ 
#ifdef PROFILING
        alloc_t2 = rdtimercpu->Stop();
        serializeTime->Serialize(rdtimercpu);
        rdtimercpu->Reset("CPU to GPU Transfer Time");
        rdtimercpu->Start();
#endif
  HIP_CHECK(hipMemcpy(hidden_delta_cuda, net->hidden_delta, (hid + 1) * sizeof(float), hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(input_prev_weights_cuda, input_weights_prev_one_dim, (in + 1) * (hid + 1) * sizeof(float), hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(input_hidden_cuda, input_weights_one_dim, (in + 1) * (hid + 1) * sizeof(float), hipMemcpyHostToDevice));

/*cpu-gpu transfer-stop, kernel exec-start */
#ifdef PROFILING
    cpu_to_gpu_t2 =  rdtimercpu->Stop();
    serializeTime->Serialize(rdtimercpu);
    rdtimercpu->Reset("COMPUTE:Kernel Execution Time");
    hipDeviceSynchronize();
    rdtimercpu->Start();
#endif

  hipLaunchKernelGGL(bpnn_adjust_weights_cuda, dim3(grid), dim3(threads ), 0, 0, hidden_delta_cuda,
												hid, 
												input_cuda, 
												in,
												input_hidden_cuda, 
												input_prev_weights_cuda
								);
  HIP_CHECK(hipGetLastError());
/* kernel exec-stop, gpu-cpu transfer-start */
#ifdef PROFILING
    kernel_t2 = rdtimercpu->Stop();
    serializeTime->Serialize(rdtimercpu);
    rdtimercpu->Reset("GPU to CPU Transfer Time");
    rdtimercpu->Start();
#endif
  HIP_CHECK(hipMemcpy(net->input_units, input_cuda, (in + 1) * sizeof(float), hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(input_weights_one_dim, input_hidden_cuda, (in + 1) * (hid + 1) * sizeof(float), hipMemcpyDeviceToHost));
/* gpu-cpu transfer- stop */ 
#ifdef PROFILING        
    gpu_to_cpu_t2= rdtimercpu->Stop();
    serializeTime->Serialize(rdtimercpu);
#endif

/* over all stop print all times delete timers */
#ifdef PROFILING    
    overall_cpu_t =  rdtimerOverallCpu->Stop();
    serializeTime->Serialize(rdtimerOverallCpu);
      printf("time CPU to GPU memory copy = %fs\n", cpu_to_gpu_t1+cpu_to_gpu_t2);
      printf("time GPU to CPU memory copy back = %fs\n", gpu_to_cpu_t1+gpu_to_cpu_t2);
      printf("time GPU malloc = %fs\n", alloc_t1+alloc_t2);
      printf("time kernel = %fs\n", (kernel_t1+kernel_t2));
      printf("Overall CPU time = %fs\n", overall_cpu_t);

          delete rdtimercpu;
          delete serializeTime;
          delete rdtimerOverallCpu;

 #endif

  FILE *result = fopen("result.txt", "w");
  if (!result) {
    fprintf(stderr, "Could not open result.txt\n");
    exit(EXIT_FAILURE);
  }
  fprintf(result, "input %d\n", in);
  for (int j = 1; j <= hid; j++)
    fprintf(result, "sum %d %.9g\n", j, raw_sum[j]);
  for (int j = 1; j <= hid; j++)
    fprintf(result, "hidden %d %.9g\n", j, net->hidden_units[j]);
  if (fclose(result) != 0) {
    fprintf(stderr, "Could not close result.txt\n");
    exit(EXIT_FAILURE);
  }

  HIP_CHECK(hipFree(input_cuda));
  HIP_CHECK(hipFree(output_hidden_cuda));
  HIP_CHECK(hipFree(input_hidden_cuda));
  HIP_CHECK(hipFree(hidden_partial_sum));
  HIP_CHECK(hipFree(input_prev_weights_cuda));
  HIP_CHECK(hipFree(hidden_delta_cuda));
  
  free(partial_sum);
  free(input_weights_one_dim);
  free(input_weights_prev_one_dim);

#endif   

}
