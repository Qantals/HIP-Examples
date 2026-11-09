#include "hip/hip_runtime.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <assert.h>

#ifdef PROFILING
#include "RDTimer.h"
#endif

#define BLOCK_SIZE 256
#define STR_SIZE 256
#define DEVICE 0
#define HALO 1 // halo width along one direction when advancing to the next iteration

//#define BENCH_PRINT
#define MIN(a, b) ((a)<=(b) ? (a) : (b))

int run(int argc, char** argv);

int rows, cols;
int* data;
int** wall;
int* result;
#define M_SEED 9
int pyramid_height;

//#define BENCH_PRINT
#ifdef PROFILING
SimplePerfSerializer* serializeTime;
RDTimerCPU* rdtimercpu;
#endif

void
init(int argc, char** argv)
{
	if(argc==4){

		cols = atoi(argv[1]);

		rows = atoi(argv[2]);

                pyramid_height=atoi(argv[3]);
	}else{
                fprintf(stderr, "Usage: pathfinder cols rows pyramid_height\n");
                exit(EXIT_FAILURE);
        }
	if (cols <= 0 || rows <= 1 || pyramid_height <= 0 ||
	    2 * pyramid_height >= BLOCK_SIZE) {
		fprintf(stderr, "Invalid Pathfinder dimensions or pyramid height\n");
		exit(EXIT_FAILURE);
	}
	data = new int[rows*cols];

	wall = new int*[rows];

	for(int n=0; n<rows; n++)

		wall[n]=data+cols*n;

	result = new int[cols];


	int seed = M_SEED;

	srand(seed);



	for (int i = 0; i < rows; i++)

    {

        for (int j = 0; j < cols; j++)

        {

            wall[i][j] = rand() % 10;

        }

    }


#ifdef BENCH_PRINT

    for (int i = 0; i < rows; i++)

    {

        for (int j = 0; j < cols; j++)

        {

            printf("%d ",wall[i][j]) ;

        }

        printf("\n") ;

    }

#endif
}

void
fatal(char *s)
{
	fprintf(stderr, "error: %s\n", s);

}

#define IN_RANGE(x, min, max)   ((x)>=(min) && (x)<=(max))
#define CLAMP_RANGE(x, min, max) x = (x<(min)) ? min : ((x>(max)) ? max : x )
#define MIN(a, b) ((a)<=(b) ? (a) : (b))

__global__ void dynproc_kernel(
	int iteration,
	int *gpuWall,
	int *gpuSrc,
	int *gpuResults,
	int cols,
	int rows,
	int startStep,
	int border)
{

        __shared__ int prev[BLOCK_SIZE];
        __shared__ int result[BLOCK_SIZE];

	int bx = hipBlockIdx_x;
	int tx=hipThreadIdx_x;

        // each block finally computes result for a small block
        // after N iterations.
        // it is the non-overlapping small blocks that cover
        // all the input data

        // calculate the small block size
	int small_block_cols = BLOCK_SIZE-iteration*HALO*2;

        // calculate the boundary for the block according to
        // the boundary of its small block
        int blkX = small_block_cols*bx-border;
        int blkXmax = blkX+BLOCK_SIZE-1;

        // calculate the global thread coordination
	int xidx = blkX+tx;

        // effective range within this block that falls within
        // the valid range of the input data
        // used to rule out computation outside the boundary.
        int validXmin = (blkX < 0) ? -blkX : 0;
        int validXmax = (blkXmax > cols-1) ? BLOCK_SIZE-1-(blkXmax-cols+1) : BLOCK_SIZE-1;

        int W = tx-1;
        int E = tx+1;

        W = (W < validXmin) ? validXmin : W;
        E = (E > validXmax) ? validXmax : E;

        bool isValid = IN_RANGE(tx, validXmin, validXmax);

	if(IN_RANGE(xidx, 0, cols-1)){
            prev[tx] = gpuSrc[xidx];
	}
	__syncthreads(); // [Ronny] Added sync to avoid race on prev Aug. 14 2012
        bool computed;
        for (int i=0; i<iteration ; i++){
            computed = false;
            if( IN_RANGE(tx, i+1, BLOCK_SIZE-i-2) &&  \
                  isValid){
                  computed = true;
                  int left = prev[W];
                  int up = prev[tx];
                  int right = prev[E];
                  int shortest = MIN(left, up);
                  shortest = MIN(shortest, right);
                  int index = cols*(startStep+i)+xidx;
                  result[tx] = shortest + gpuWall[index];

            }
            __syncthreads();
            if(i==iteration-1)
                break;
            if(computed)	 //Assign the computation range
                prev[tx]= result[tx];
	    __syncthreads(); // [Ronny] Added sync to avoid race on prev Aug. 14 2012
      }

      // update the global memory
      // after the last iteration, only threads coordinated within the
      // small block perform the calculation and switch on ``computed''
      if (computed){
          gpuResults[xidx]=result[tx];
      }
}
/*
   compute N time steps
*/
int calc_path(int *gpuWall, int *gpuResult[2], int rows, int cols, \
	 int pyramid_height, int blockCols, int borderCols)
{
        dim3 dimBlock(BLOCK_SIZE);
        dim3 dimGrid(blockCols);

        int src = 1, dst = 0;
	for (int t = 0; t < rows-1; t+=pyramid_height) {
            int temp = src;
            src = dst;
            dst = temp;
            hipLaunchKernelGGL(dynproc_kernel, dim3(dimGrid), dim3(dimBlock), 0, 0,
                MIN(pyramid_height, rows-t-1),
                gpuWall, gpuResult[src], gpuResult[dst],
                cols,rows, t, borderCols);
	    if (hipGetLastError() != hipSuccess) {
	        fprintf(stderr, "Pathfinder HIP kernel launch failed\n");
	        return -1;
	    }
	}
        return dst;
}

int main(int argc, char** argv)
{
    int num_devices;

#ifdef PROFILING
	float overall_cpu_t;
	RDTimerCPU* rdtimerOverallCpu = new RDTimerCPU();
	rdtimerOverallCpu->Reset("Overall CPU Time");
	serializeTime = new SimplePerfSerializer( argv[0] );
	rdtimercpu = new RDTimerCPU();
	rdtimerOverallCpu->Start();
#endif
    if (hipGetDeviceCount(&num_devices) != hipSuccess || num_devices < 1 ||
        hipSetDevice(DEVICE) != hipSuccess) {
        fprintf(stderr, "No usable HIP device\n");
        return EXIT_FAILURE;
    }

    int retVal= run(argc,argv);

#ifdef PROFILING
	overall_cpu_t = rdtimerOverallCpu->Stop();
	serializeTime->Serialize(rdtimerOverallCpu);

	//printf("Overall CPU time = %fs\n", overall_cpu_t);

	delete rdtimercpu;
	delete rdtimerOverallCpu;
	delete serializeTime;
 #endif
    return retVal;
}

int run(int argc, char** argv)
{
    init(argc, argv);

    /* --------------- pyramid parameters --------------- */
    int borderCols = (pyramid_height)*HALO;
    int smallBlockCol = BLOCK_SIZE-(pyramid_height)*HALO*2;
    int blockCols = cols/smallBlockCol+((cols%smallBlockCol==0)?0:1);

    printf("pyramidHeight: %d\ngridSize: [%d]\nborder:[%d]\nblockSize: %d\nblockGrid:[%d]\ntargetBlock:[%d]\n",\
	pyramid_height, cols, borderCols, BLOCK_SIZE, blockCols, smallBlockCol);

    int *gpuWall, *gpuResult[2];
    int size = rows*cols;
#ifdef PROFILING
	rdtimercpu->Reset("Malloc Time");
	rdtimercpu->Start();
#endif
    if (hipMalloc((void**)&gpuResult[0], sizeof(int)*cols) != hipSuccess ||
        hipMalloc((void**)&gpuResult[1], sizeof(int)*cols) != hipSuccess ||
        hipMalloc((void**)&gpuWall, sizeof(int)*(size-cols)) != hipSuccess) {
        fprintf(stderr, "Pathfinder HIP allocation failed\n");
        return EXIT_FAILURE;
    }
#ifdef PROFILING
    rdtimercpu->Stop();
    serializeTime->Serialize(rdtimercpu);
    rdtimercpu->Reset("CPU to GPU Transfer Time");
    rdtimercpu->Start();
#endif
    if (hipMemcpy(gpuResult[0], data, sizeof(int)*cols, hipMemcpyHostToDevice) != hipSuccess ||
        hipMemcpy(gpuWall, data+cols, sizeof(int)*(size-cols), hipMemcpyHostToDevice) != hipSuccess) {
        fprintf(stderr, "Pathfinder HIP input transfer failed\n");
        return EXIT_FAILURE;
    }
#ifdef PROFILING
    rdtimercpu->Stop();
    serializeTime->Serialize(rdtimercpu);
    rdtimercpu->Reset("COMPUTE:Compute Time");
    rdtimercpu->Start();
#endif
    int final_ret = calc_path(gpuWall, gpuResult, rows, cols, \
	 pyramid_height, blockCols, borderCols);
    if (final_ret < 0) return EXIT_FAILURE;
#ifdef PROFILING
    rdtimercpu->Stop();
    serializeTime->Serialize(rdtimercpu);
    rdtimercpu->Reset("GPU to CPU Transfer Time");
    rdtimercpu->Start();
#endif
    if (hipDeviceSynchronize() != hipSuccess ||
        hipMemcpy(result, gpuResult[final_ret], sizeof(int)*cols, hipMemcpyDeviceToHost) != hipSuccess) {
        fprintf(stderr, "Pathfinder HIP execution or output transfer failed\n");
        return EXIT_FAILURE;
    }
#ifdef PROFILING
    rdtimercpu->Stop();
    serializeTime->Serialize(rdtimercpu);
#endif
	FILE *output = fopen("result.txt", "w");
	if (output == NULL) {
		perror("result.txt");
		return EXIT_FAILURE;
	}
	for (int i = 0; i < cols; i++)
		fprintf(output, "%d\n", result[i]);
	if (fclose(output) != 0) {
		perror("result.txt");
		return EXIT_FAILURE;
	}

#ifdef BENCH_PRINT

    for (int i = 0; i < cols; i++)

            printf("%d ",data[i]) ;

    printf("\n") ;

    for (int i = 0; i < cols; i++)

            printf("%d ",result[i]) ;

    printf("\n") ;

#endif


    hipFree(gpuWall);
    hipFree(gpuResult[0]);
    hipFree(gpuResult[1]);

    delete [] data;
    delete [] wall;
    delete [] result;

	return EXIT_SUCCESS;
}
