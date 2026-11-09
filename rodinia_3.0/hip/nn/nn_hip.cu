#include "hip/hip_runtime.h"
/*
 * nn.cu
 * Nearest Neighbor
 *
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <float.h>
#include <vector>
#ifdef __CUDA_CC__
#include "cuda.h"
#endif
#include <math.h>
#define min( a, b )			((a) > (b) ? (b) : (a))
#define ceilDiv( a, b )		(((a) + (b) - 1) / (b))
#define print( x )			printf( #x ": %lu\n", (unsigned long) x )
#define DEBUG false		

#define DEFAULT_THREADS_PER_BLOCK 256

#define MAX_ARGS 10
#define REC_LENGTH 53 // size of a record in db
#define LATITUDE_POS 28	// character position of the latitude value in each record
#define OPEN 10000	// initial value of nearest neighbors

#define HIP_CHECK(call)                                                        \
  do {                                                                         \
    hipError_t error = (call);                                                  \
    if (error != hipSuccess) {                                                  \
      fprintf(stderr, "HIP error at %s:%d: %s\n", __FILE__, __LINE__,          \
              hipGetErrorString(error));                                        \
      exit(EXIT_FAILURE);                                                       \
    }                                                                           \
  } while (0)

typedef struct latLong
{
  float lat;
  float lng;
} LatLong;

typedef struct record
{
  char recString[REC_LENGTH];
  float distance;
} Record;

int loadData(char *filename,std::vector<Record> &records,std::vector<LatLong> &locations);
void findLowest(std::vector<Record> &records,float *distances,int numRecords,int topN);
void printUsage();
int parseCommandline(int argc, char *argv[], char* filename,int *r,float *lat,float *lng,
                     int *q, int *t, int *p, int *d);

/**
* Kernel
* Executed on GPU
* Calculates the Euclidean distance from each record in the database to the target position
*/

__global__ void euclid(LatLong *d_locations, float *d_distances, int numRecords,float lat, float lng)
{
	//int globalId = hipGridDim_x * hipBlockDim_x * hipBlockIdx_y + hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
	int globalId = hipBlockDim_x * ( hipGridDim_x * hipBlockIdx_y + hipBlockIdx_x ) + hipThreadIdx_x; // more efficient
    LatLong *latLong = d_locations+globalId;
    if (globalId < numRecords) {
        float *dist=d_distances+globalId;
        *dist = (float)sqrt((lat-latLong->lat)*(lat-latLong->lat)+(lng-latLong->lng)*(lng-latLong->lng));
	}
}
/**
* This program finds the k-nearest neighbors
**/

int main(int argc, char* argv[])
{
	int    i=0;
	float lat=0.0f, lng=0.0f;
	int quiet=0,timing=0,platform=-1,device=-1;
	int verifiedResult = 0;
    std::vector<Record> records;
	std::vector<LatLong> locations;
	char filename[100];
	int resultsCount=10;
    // parse command line
    if (parseCommandline(argc, argv, filename,&resultsCount,&lat,&lng,
                     &quiet, &timing, &platform, &device)) {
      printUsage();
      return EXIT_FAILURE;
    }
	(void)timing;
	(void)platform;

    int numRecords = loadData(filename,records,locations);
    if (numRecords <= 0) {
      fprintf(stderr, "Error: no database records were loaded.\n");
      return EXIT_FAILURE;
    }
    if (resultsCount > numRecords) resultsCount = numRecords;

    //for(i=0;i<numRecords;i++)
    //  printf("%s, %f, %f\n",(records[i].recString),locations[i].lat,locations[i].lng);


    //Pointers to host memory
	float *distances;
	float *distancesRef;
	//Pointers to device memory
	LatLong *d_locations;
	float *d_distances;


	// Scaling calculations - added by Sam Kauffman
	hipDeviceProp_t deviceProp;
	if (device >= 0) HIP_CHECK(hipSetDevice(device));
	HIP_CHECK(hipGetDeviceProperties(&deviceProp, device >= 0 ? device : 0));
	HIP_CHECK(hipDeviceSynchronize());

	unsigned long  maxGridX = deviceProp.maxGridSize[0];
        //printf("maxGridX : %lu \n",maxGridX );
	unsigned long threadsPerBlock = min( deviceProp.maxThreadsPerBlock, DEFAULT_THREADS_PER_BLOCK );
	size_t totalDeviceMemory;
	size_t freeDeviceMemory;
	HIP_CHECK(hipMemGetInfo(&freeDeviceMemory, &totalDeviceMemory));
	HIP_CHECK(hipDeviceSynchronize());

	unsigned long usableDeviceMemory = freeDeviceMemory * 85 / 100; // 85% arbitrary throttle to compensate for known CUDA bug
	unsigned long maxThreads = usableDeviceMemory / 12; // 4 bytes in 3 vectors per thread
	if ( numRecords > maxThreads )
	{
		fprintf( stderr, "Error: Input too large.\n" );
		return EXIT_FAILURE;
	}

	unsigned long blocks = ceilDiv( numRecords, threadsPerBlock ); // extra threads will do nothing

	unsigned long gridY = ceilDiv( blocks, maxGridX );

	unsigned long gridX = ceilDiv( blocks, gridY );
	// There will be no more than (gridY - 1) extra blocks

	dim3 gridDim( gridX, gridY );
	
	if ( DEBUG )
		{
			print( totalDeviceMemory ); // 804454400
			print( freeDeviceMemory );
			print( usableDeviceMemory );
			print( maxGridX ); // 65535
			print( deviceProp.maxThreadsPerBlock ); // 1024
			print( threadsPerBlock );
			print( maxThreads );
			print( blocks ); // 130933
			print( gridY );
			print( gridX );
	}
	
	/**
	* Allocate memory on host and device
	*/
	distances = (float *)malloc(sizeof(float) * numRecords);
	if (!distances) {
		fprintf(stderr, "Error: could not allocate host distance array.\n");
		return EXIT_FAILURE;
	}
	HIP_CHECK(hipMalloc((void **) &d_locations,sizeof(LatLong) * numRecords));
	HIP_CHECK(hipMalloc((void **) &d_distances,sizeof(float) * numRecords));

	distancesRef = (float *)malloc(sizeof(float) * numRecords);
	if (!distancesRef) {
		fprintf(stderr, "Error: could not allocate reference distance array.\n");
		return EXIT_FAILURE;
	}
   /**
    * Transfer data from host to device
    */
    HIP_CHECK(hipMemcpy(d_locations, locations.data(),
                        sizeof(LatLong) * numRecords, hipMemcpyHostToDevice));
    /**
    * Execute kernel
    */

    hipLaunchKernelGGL(euclid, dim3(gridDim), dim3(threadsPerBlock ), 0, 0, d_locations,d_distances,numRecords,lat,lng);
    HIP_CHECK(hipGetLastError());
	HIP_CHECK(hipDeviceSynchronize());
    //Copy data from device memory to host memory
    HIP_CHECK(hipMemcpy(distances, d_distances, sizeof(float) * numRecords,
                        hipMemcpyDeviceToHost));

	// Rodinia 3.0 checks every GPU result against the CPU calculation.
	for( i = 0 ; i < numRecords ; i++ )
	{
		distancesRef[i] = (float)sqrt((lat-locations[i].lat)*(lat-locations[i].lat)+(lng-locations[i].lng)*(lng-locations[i].lng));
	}

	for( i = 0 ; i < numRecords ; i++ )
	{
		if (((distancesRef[i] - distances[i])*(distancesRef[i] - distances[i])) > 0.001)
		{
			fprintf(stderr, "Failed results %f   %f\n", distancesRef[i], distances[i]);
			verifiedResult = 1;
			break;
		}
	}

	// find the resultsCount least distances
    findLowest(records,distances,numRecords,resultsCount);

    // print out results
    if (!quiet)
    for(i=0;i<resultsCount;i++) {
      printf("%s --> Distance=%f\n",records[i].recString,records[i].distance);
    }

	free(distances);
	free(distancesRef);
    //Free memory
	HIP_CHECK(hipFree(d_locations));
	HIP_CHECK(hipFree(d_distances));
	return verifiedResult;
}

int loadData(char *filename,std::vector<Record> &records,std::vector<LatLong> &locations){
    FILE   *flist,*fp;
	int    i=0;
	char dbname[64];
	int recNum=0;

    /**Main processing **/

    flist = fopen(filename, "r");
	if (!flist) {
		fprintf(stderr, "Error: could not open file list '%s'.\n", filename);
		return -1;
	}
	while(fscanf(flist, "%63s", dbname) == 1) {
		/**
		* Read in all records of length REC_LENGTH
		* If this is the last file in the filelist, then done
		* else open next file to be read next iteration
		*/
        fp = fopen(dbname, "r");
        if(!fp) {
			fprintf(stderr, "Error: could not open database '%s'.\n", dbname);
			fclose(flist);
			return -1;
        }
        // read each record
		while(true){
            Record record;
            LatLong latLong;
			if (!fgets(record.recString, sizeof(record.recString), fp)) break;
			record.recString[strcspn(record.recString, "\r\n")] = '\0';
			if (strlen(record.recString) < LATITUDE_POS + 10) {
				fprintf(stderr, "Error: malformed record in '%s'.\n", dbname);
				fclose(fp);
				fclose(flist);
				return -1;
			}

            // parse for lat and long
            char substr[6];

            for(i=0;i<5;i++) substr[i] = *(record.recString+i+28);
            substr[5] = '\0';
            latLong.lat = atof(substr);

            for(i=0;i<5;i++) substr[i] = *(record.recString+i+33);
            substr[5] = '\0';
            latLong.lng = atof(substr);

            locations.push_back(latLong);
            records.push_back(record);
            recNum++;
        }
        fclose(fp);
    }
    fclose(flist);
//    for(i=0;i<rec_count*REC_LENGTH;i++) printf("%c",sandbox[i]);
    return recNum;
}

void findLowest(std::vector<Record> &records,float *distances,int numRecords,int topN){
  int i,j;
  float val;
  int minLoc;
  Record tempRec;
  float tempDist;

  for(i=0;i<topN;i++) {
    minLoc = i;
    for(j=i;j<numRecords;j++) {
      val = distances[j];
      if (val < distances[minLoc]) minLoc = j;
    }
    // swap locations and distances
    tempRec = records[i];
    records[i] = records[minLoc];
    records[minLoc] = tempRec;

    tempDist = distances[i];
    distances[i] = distances[minLoc];
    distances[minLoc] = tempDist;

    // add distance to the min we just found
    records[i].distance = distances[i];
  }
}

int parseCommandline(int argc, char *argv[], char* filename,int *r,float *lat,float *lng,
                     int *q, int *t, int *p, int *d){
    int i;
    if (argc < 2) return 1; // error
    snprintf(filename, 100, "%s", argv[1]);

    for(i=2;i<argc;i++) {
      if (argv[i][0]=='-') {// flag
		if (!strcmp(argv[i], "-h") || !strcmp(argv[i], "--help")) return 1;
		if (!strcmp(argv[i], "-q")) { *q = 1; continue; }
		if (!strcmp(argv[i], "-t")) { *t = 1; continue; }
		if (i + 1 >= argc) return 1;
		if (!strcmp(argv[i], "-r")) *r = atoi(argv[++i]);
		else if (!strcmp(argv[i], "-lat")) *lat = atof(argv[++i]);
		else if (!strcmp(argv[i], "-lng")) *lng = atof(argv[++i]);
		else if (!strcmp(argv[i], "-p")) *p = atoi(argv[++i]);
		else if (!strcmp(argv[i], "-d")) *d = atoi(argv[++i]);
		else return 1;
      }
	  else return 1;
    }
	if (*r <= 0) return 1;
    if ((*d >= 0 && *p<0) || (*p>=0 && *d<0)) // both p and d must be specified if either are specified
      return 1;
    return 0;
}

void printUsage(){
  printf("Nearest Neighbor Usage\n");
  printf("\n");
  printf("nearestNeighbor [filename] -r [int] -lat [float] -lng [float] [-hqt] [-p [int] -d [int]]\n");
  printf("\n");
  printf("example:\n");
  printf("$ ./nearestNeighbor filelist.txt -r 5 -lat 30 -lng 90\n");
  printf("\n");
  printf("filename     the filename that lists the data input files\n");
  printf("-r [int]     the number of records to return (default: 10)\n");
  printf("-lat [float] the latitude for nearest neighbors (default: 0)\n");
  printf("-lng [float] the longitude for nearest neighbors (default: 0)\n");
  printf("\n");
  printf("-h, --help   Display the help file\n");
  printf("-q           Quiet mode. Suppress all text output.\n");
  printf("-t           Print timing information.\n");
  printf("\n");
  printf("-p [int]     Choose the platform (must choose both platform and device)\n");
  printf("-d [int]     Choose the device (must choose both platform and device)\n");
  printf("\n");
  printf("\n");
  printf("Notes: 1. The filename is required as the first parameter.\n");
  printf("       2. If you declare either the device or the platform,\n");
  printf("          you must declare both.\n\n");
}
