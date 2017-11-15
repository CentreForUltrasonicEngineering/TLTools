/*******

Developed by Timothy Lardner and Jerzy Dziewierz,
Centre for Ultrasonic Engineering,
University of Strathclyde,
Copyright 2017,

Not to be used or copied without authorization from the authors,
This superseeds any other licence that may be bundled with this software.

For more information see:
"Inspection of complex components using 2D arrays and TFM"
"A design methodology for 2D sparse NDE arrays  using an efficient implementation of refracted - ray TFM"

http://www.cue.ac.uk

*******/


#define COEFFGEN_nTimePointsPerLine 33  // 17 points in a line for calculation, 16 for checking the polyfit
#define classicMinSearch_spread 1e-3
#define COEFF_SIZE 5
#define COEFFGEN_nTimePointsPerToFit 17
#define COEFFGEN_nTimePointsPerToCheck 16

struct SurfParam
{
    float x1, y1, z1, x2, y2, z2;    // position of source and target point
    float slow1, slow2;              // slowness of wave in medium 1 and 2
    float c0, c1;// c2, c3, c4, c5, c6, c7, c8, c9, c10, c11, c12, c13, c14; // for ny,y0,dy,nx,x0,dx
  //  float y0limit, y1limit;
    float *DataVector;                  // can be used as a pointer to a longer surface descriptor
    int DataVectorElementCount;         // can be used as a pointer to a longer surface descriptor
};

//static SurfParam *GlobalParam = new SurfParam();

__global__  void TestStruct1(float* output,SurfParam* parameters){
    int idx=threadIdx.x+blockDim.x*blockIdx.x;
    output[idx] = parameters->slow1;
}


__device__ __host__ void polyfit17x5(double output[5], double A[34]);
__device__ __host__ inline float PerItem_TOF(SurfParam& Params);
__device__ __host__ inline float PerItem_TOF_CF(float &x, float &y, SurfParam& Params);
__device__ __host__ inline float PerItem(float &x,float &y, SurfParam& Params) ;
__device__ __host__ inline float FlatZ_TOF(SurfParam& Params);
__device__ __host__ inline float FlatZ_TOF_CF(float &x, float &y, SurfParam& Params);
__device__ __host__ inline float FlatZ(float &x,float &y, SurfParam& Params) ;
__device__ __host__ inline void sortArray(float &fv1, float &fv2, float &fv3, float &vx1, float &vy1, float &vx2, float &vy2, float &vx3, float &vy3);
__device__ __host__ inline void swapArray(float &a1,float &a2, float &a3,float &b1,float &b2,float &b3);
__device__ __host__ float classicMinSearch(float (*CostFunction)(float& ,float& ,SurfParam&), SurfParam& Params,float &vx0,float &vy0);
__host__ __device__ inline void ind2sub(int *siz, int idx, int *sub);
__device__ __host__ inline double Square(double &x);
__device__ __host__ inline double Reciprocal( double &x);
__device__ __host__ inline double Power3(double &x);
__device__ __host__ inline double Power4(double &x);



__global__ void transform_tpoints_into_coeffs2GPU_kernel(
        int total_coefflines,
        int kernel_ProbeElementCount,
        int kernel_ny,
        const float* ZVector,
        const float* TBuffer,
        float* CoeffBuffer
        )
{ // per-coeffset kernel
    int idx_coeffline=blockIdx.x * blockDim.x + threadIdx.x;
    if (idx_coeffline>total_coefflines) {return;}


            // convert idx_coeffline to idx_tx and idx_zline using ind2sub:
        int size[4];
        int subs[4];
        size[3]=kernel_ProbeElementCount;
        size[2]=kernel_ny;
        size[1]=1;
        size[0]=1;
        ind2sub(size,idx_coeffline,subs);
        int idx_tx=subs[3];
        int idx_zline=subs[2];

        // calculate buffer offsets:
        int CoefflineOffset=idx_coeffline*COEFF_SIZE;
        int ZVector_offset=0;

        int TBuffer_offset=(idx_tx+idx_zline*kernel_ProbeElementCount)*COEFFGEN_nTimePointsPerLine;

        //idx_NaNBuffer=0+2*(idx_tx+persistent_ProbeElementCount*(idx_zline));// not used here
        // local buffers, double precision for polyfit 17x5
        double fitinput[34];
        double polycoeff5[5];

        // fill the local buffers
        // NOTE: format for 'fitinput' is float [x0,x1,...,x16,y0,y1,...,y16] - this comes compiled from Mathematica
        // put tbase first
        //if (persistent_verbosemode){mexPrintf("tbase:\n");};
        for (int idx_tbase=0;idx_tbase<COEFFGEN_nTimePointsPerToFit; idx_tbase++)
        {
            fitinput[idx_tbase]=(double)(ZVector[2*idx_tbase+ZVector_offset]); // !note, every 2nd z-sample is taken for fit source table
            //if (persistent_verbosemode){mexPrintf("fitinput[%d]=%e\n",idx_tbase,fitinput[idx_tbase]);};
        }
        // THIS loads points from TBuffer to temporary, double-precision buffer suitable for PolyFit
        // put every 2nd timepoint now
        for (int idx_tbase=0;idx_tbase<COEFFGEN_nTimePointsPerToFit; idx_tbase++)
        {
            fitinput[COEFFGEN_nTimePointsPerToFit+idx_tbase]=(double)(TBuffer[TBuffer_offset+2*idx_tbase]); // !note, every 2nd z-sample is taken for fit source table
        }
        // transform the fit input into coeff buffer using the Mathematica-generated code
        polyfit17x5(polycoeff5,fitinput);
        // load the items from polycoeff5 to CoeffBuffer
        CoeffBuffer[CoefflineOffset+COEFF_SIZE-1]=(float)polycoeff5[0];
        CoeffBuffer[CoefflineOffset+COEFF_SIZE-2]=(float)polycoeff5[1];
        CoeffBuffer[CoefflineOffset+COEFF_SIZE-3]=(float)polycoeff5[2];
        CoeffBuffer[CoefflineOffset+COEFF_SIZE-4]=(float)polycoeff5[3];
        CoeffBuffer[CoefflineOffset+COEFF_SIZE-5]=(float)polycoeff5[4];
}

__global__ void GenerateTimePoints(float* TimeBuffer,float* ZVector, float* ydim, float* ProbeLocation, int SurfID, int n_elem, int ny, int NoOfTimePoints,SurfParam* parameters){

    int idx=threadIdx.x+blockDim.x*blockIdx.x; // Only launch 1D block/grid
    if (idx>NoOfTimePoints){
        return; // If we launch more threads than we have timepoints (and we shouldn't), don't run
    }


    int subs[4];
    int size[4];
    size[3] = (int)COEFFGEN_nTimePointsPerLine;
    size[2] = (int)n_elem; // Number of elements in the probe
    size[1] = (int)ny; // Number of pixels in the y-direction
    size[0] = 1;

    ind2sub(size,idx,subs);
    int yline_idx=subs[1];
    int probeElementIdx=subs[2];
    int timePointOfLine=subs[3];



    float probe_y=ProbeLocation[1+3*probeElementIdx];  // Y-Point of the probe element
    float probe_z=ProbeLocation[2+3*probeElementIdx];  // Z-Point of the probe element
    float yy=ydim[yline_idx];                          // Y-Point that this kernel is calculating for
    float zz=ZVector[timePointOfLine];          // Z-Point that this kernel is calculating for

    float tof; // The final time of flight



    SurfParam localParam = *parameters; // Create a local copy of the parameters for each thread
    localParam.y1 = probe_y;
    localParam.z1=probe_z;
    localParam.y2=yy;
    localParam.z2=zz;

    if(SurfID==0){
        // No refraction takes place
        float dy=probe_y-yy;
        float dz=probe_z-zz;
        tof=sqrtf(dy*dy+dz*dz)*localParam.slow1; // We invert the speed once as multiplying is faster than dividing
        // note that this ignores X-coordinates completely! hence, no X-correction is applied whatsoever!
    }
    else if(SurfID==1){
        // FlatZ refraction takes place
        tof=FlatZ_TOF(localParam);
    }
    else if(SurfID==2){
        // 2D-Surface refraction takes place
        tof=PerItem_TOF(localParam);
    }
    TimeBuffer[idx]=tof;
}

__global__ void TFM_coeff(float *dest, float *FMC, float *Elem, int n_elem, float fs, float *zdim, int nz, int sample_length, float time_start,float* Coeffs)
{
  int idx=threadIdx.x+blockDim.x*blockIdx.x;

  float tx_path;
  float rx_path;
  float time;
  float accumulator = 0;
  int sample_truncated;
  float sample_weight;
  int z_location = idx/nz;
  int y_location = idx%nz;
  int final_sample;
  float z_position = zdim[z_location];
  int tx_coeff_offset;
  int rx_coeff_offset;

  for(int i=0; i<n_elem; i++){

      tx_coeff_offset = 5*y_location*n_elem + 5*i;
      tx_path = Coeffs[tx_coeff_offset];
      tx_path = z_position*tx_path + Coeffs[tx_coeff_offset+1];
      tx_path = z_position*tx_path + Coeffs[tx_coeff_offset+2];
      tx_path = z_position*tx_path + Coeffs[tx_coeff_offset+3];
      tx_path = z_position*tx_path + Coeffs[tx_coeff_offset+4];
      for(int j=0; j<n_elem;j++){
         rx_coeff_offset = 5*y_location*n_elem + 5*j;
         rx_path = Coeffs[rx_coeff_offset];
         rx_path = z_position*rx_path + Coeffs[rx_coeff_offset+1];
         rx_path = z_position*rx_path + Coeffs[rx_coeff_offset+2];
         rx_path = z_position*rx_path + Coeffs[rx_coeff_offset+3];
         rx_path = z_position*rx_path + Coeffs[rx_coeff_offset+4];
         sample_truncated = floorf(fminf(fmaxf(((tx_path+rx_path)- time_start)*fs,0.0f),sample_length-2.0f));
         sample_weight = fminf(fmaxf(time*fs-sample_truncated,0.0f),1.0f);
         final_sample = (i*n_elem+j)*sample_length + sample_truncated;
         accumulator = accumulator + (1.0-sample_weight)*FMC[final_sample] + sample_weight*FMC[final_sample+1];
      }
  }
  dest[idx] = accumulator;
  //dest[idx] = 1;
  
}

__global__ void TFM(float *dest, float *FMC, float *Elem, int n_elem, float *ydim, float* zdim, float speed, int ny, int nz,float fs,int sample_length, float time_start)
{
  int idx=threadIdx.x+blockDim.x*blockIdx.x;
  float tx_location;
  float rx_location;
  float tx_depth;
  float rx_depth;
  float tx_path;
  float rx_path;
  float time;
  float accumulator = 0;
  int sample_truncated;
  float sample_weight;
  int z_location = idx/nz;
  int y_location = idx%nz;
  int final_sample;

  for(int i=0; i<n_elem; i++){
      tx_location = Elem[3*i+1];
      tx_depth = Elem[3*i+2];
      tx_path = sqrt((tx_location-ydim[y_location])*(tx_location-ydim[y_location]) + (tx_depth-zdim[z_location])*(tx_depth-zdim[z_location]));

      for(int j=0; j<n_elem;j++){
         rx_location = Elem[3*j+1];
         rx_depth = Elem[3*j+2];
         rx_path = sqrt((rx_location-ydim[y_location])*(rx_location-ydim[y_location]) + (rx_depth-zdim[z_location])*(rx_depth-zdim[z_location]));
         sample_truncated = floorf(fminf(fmaxf(((tx_path+rx_path) / speed - time_start)*fs,0.0f),sample_length-2.0f));
         sample_weight = fminf(fmaxf(time*fs-sample_truncated,0.0f),1.0f);
         final_sample = (i*n_elem+j)*sample_length + sample_truncated;
         accumulator = accumulator + (1.0-sample_weight)*FMC[final_sample] + sample_weight*FMC[final_sample+1];
      }
  }
  //dest[idx] = rx_path/speed;
  dest[idx] = accumulator;
}

__device__ __host__ inline float PerItem_TOF(SurfParam& Params)
{

    float x0_guess=0.5*(Params.x1+Params.x2);
    float y0_guess=0.5*(Params.y1+Params.y2);
    float tof = classicMinSearch(&PerItem_TOF_CF,Params,x0_guess,y0_guess);
    return tof;
}

__device__ __host__ inline float PerItem_TOF_CF(float &x, float &y, SurfParam& Params)
{
    float surfaceZ=PerItem(x,y,Params);
    float dx1=Params.x1-x;
    float dy1=Params.y1-y;
    float dz1=Params.z1-surfaceZ;
    float dx2=x-Params.x2;
    float dy2=y-Params.y2;
    float dz2=surfaceZ-Params.z2;
    float tof=sqrtf(dx1*dx1+dy1*dy1+dz1*dz1)*Params.slow1+sqrtf(dx2*dx2+dy2*dy2+dz2*dz2)*Params.slow2;
    return tof;
}

__device__ __host__ inline float PerItem(float &x,float &y, SurfParam& Params)
{
    float pickupPointerF=fmaxf(0,fminf((float)Params.DataVectorElementCount-1,((y-Params.c0)/Params.c1)));
    float pickupPointerBase=floorf(pickupPointerF);
    float pickupPointerRem=pickupPointerF-pickupPointerBase; // use this to blend samples
    int pickupPointerBaseI=(int)pickupPointerBase;
    float result = 0;
    result=(1-pickupPointerRem)*Params.DataVector[pickupPointerBaseI]+(pickupPointerRem)*Params.DataVector[pickupPointerBaseI+1];
    return result;
}

__device__ __host__ inline float FlatZ_TOF(SurfParam& Params)
{
    float x0_guess=0.5*(Params.x1+Params.x2);
    float y0_guess=0.5*(Params.y1+Params.y2);
    float tof = classicMinSearch(&FlatZ_TOF_CF,Params,x0_guess,y0_guess);
    return tof;
}

__device__ __host__ inline float FlatZ_TOF_CF(float &x, float &y, SurfParam& Params)
{
   float surfaceZ=FlatZ(x,y,Params);
   float dx1=Params.x1-x;
   float dy1=Params.y1-y;
   float dz1=Params.z1-surfaceZ;
   float dx2=x-Params.x2;
   float dy2=y-Params.y2;
   float dz2=surfaceZ-Params.z2;
   float tof=sqrtf(dx1*dx1+dy1*dy1+dz1*dz1)*Params.slow1+sqrtf(dx2*dx2+dy2*dy2+dz2*dz2)*Params.slow2;
   return tof;
}

__device__ __host__ inline float FlatZ(float &x,float &y, SurfParam& Params)
{
    return float(0);
}

__host__ __device__ inline void ind2sub(int *siz, int idx, int *sub)
{
int prod[4];
                prod[0] = siz[3]*siz[2]*siz[1];
                prod[1] = siz[3]*siz[2];
                prod[2] = siz[3];
                prod[3] = 1;
                sub[0] = (int)floor(    (float)idx / prod[0]                                                    );
                sub[1] = (int)floor(    (float)(        idx % prod[0]   )/prod[1]                               );
                sub[2] = (int)floor( (float)( ( idx % prod[0]   )%prod[1]       )  / prod[2]);
                sub[3] =                ( (     idx % prod[0]   )%prod[1]       )  % prod[2] ;
}

__device__ __host__ float classicMinSearch(float (*CostFunction)(float&, float& ,SurfParam&), SurfParam& Params,float &vx0, float &vy0)
{

    #define tolf 1e-10

//! Exit condition, tolerance on X,Y value
    #define tolx 1e-5

//! Simplex method parameter, controls the way the problem space is explored
    #define rho 1

//! Simplex method parameter, controls the way the problem space is explored
    #define chi 2

//! Simplex method parameter, controls the way the problem space is explored
    #define psi 0.5

//! Simplex method parameter, controls the way the problem space is explored
    #define sigma 0.5

// Maximum number of iterations for Nedlder-Mead search
    #define ITERATION_LIMIT 10000


    int how = 0; // mode of exploration, changes over iterations.

    // get the initial simplex CF values:
    float vx1 = vx0;
    float vy1 = vy0;
    float fv1 = (*CostFunction)(vx1,vy1,Params);

    float vx2 = vx0+classicMinSearch_spread;
    float vy2 = vy0;
    float fv2 = (*CostFunction)(vx2,vy2,Params);

    float vx3 = vx0;
    float vy3 = vy0+classicMinSearch_spread;
    float fv3 = (*CostFunction)(vx3,vy3,Params);

    // sort the CF points:
    sortArray(fv1,fv2,fv3,vx1,vy1,vx2,vy2,vx3,vy3);



    int loops = 1;

    // begin the exploration:
    while((abs(fv2-fv1) > tolf)|(abs(vx1-vx2)>tolx)|(abs(vy1-vy2)>tolx))
    {
        loops++;
        if(loops>ITERATION_LIMIT){
            return (float)fv1; // return best approximation
        }

        // exploration pivot point:
        float xbar = (vx1 + vx2)/2;
        float ybar = (vy1 + vy2)/2;
        // propose new point:
        float xr = (1 + rho)*xbar - rho*vx3;
        float yr = (1 + rho)*ybar - rho*vy3;
        float fxr = (*CostFunction)(xr,yr,Params);
        // test new point:
        if(fxr<fv1){
            // fail. make a proposal 2:
            float xe = (1 + rho*chi)*xbar - rho*chi*vx3;
            float ye = (1 + rho*chi)*ybar - rho*chi*vy3;
            float fxe = (*CostFunction)(xe,ye,Params);
            // test 2nd proposal:
            if (fxe < fxr){
                // success. Replace the worst point with proposal 2 point.
                vx3 = xe;
                vy3 = ye;
                fv3 = fxe;
            }
            else{
                // success. Replace the worst point with the proposal 1 point.
                vx3 = xr;
                vy3 = yr;
                fv3 = fxr;}
        }
        else{
            // a better success: proposal 1 is better than point2 on the simplex.
            if (fxr < fv2){
                vx3 = xr;
                vy3 = yr;
                fv3 = fxr;
            }
            else{
                if(fxr < fv3){
                    // make proposal 3:
                    float xc = (1 + psi*rho)*xbar - psi*rho*vx3;
                    float yc = (1 + psi*rho)*ybar - psi*rho*vy3;
                    float fxc = (*CostFunction)(xc,yc,Params);
                    if(fxc <=fxr){
                        // success. Adopt the proposal 3.
                        vx3 = xc;
                        vy3 = yc;
                        fv3 = fxc;
                        how = 0;
                    }
                    else{
                        how = 1;
                    }
                }
                else{
                    // make proposal 4:
                    float xcc = (1-psi)*xbar + psi*vx3;
                    float ycc = (1-psi)*ybar + psi*vx3;
                    double fxcc = (*CostFunction)(xcc,ycc,Params);
                    if(fxcc<fv3){
                        //success. Adopt the proposal 4.
                        vx3 = xcc;
                        vy3 = ycc;
                        fv3 = fxcc;
                        how = 0;
                    }
                    else{
                        how = 1;
                    }

                }
                if(how){
                    // if all the proposals failed, expand the search area:
                    vx2 = vx1 + sigma*(vx2 - vx1);
                    vy2 = vy1 + sigma*(vy2 - vy1);
                    fv2 = (*CostFunction)(vx2,vy2,Params);

                    vx3 = vx1 + sigma*(vx3 - vx1);
                    vy3 = vy1 + sigma*(vy3 - vy1);
                    fv3 = (*CostFunction)(vx3,vy3,Params);
                }

            }

        }
    // make sure that the best point is in position 1 of the simplex:
    sortArray(fv1, fv2, fv3, vx1, vy1, vx2, vy2, vx3, vy3);
    }
    vx0=vx1;
    vy0=vy1;
    return fv1;
}

// this is an ultra-simple sorting function, suitable for parallel processors. Takes 3 points and their associated metadata, and sorts.
// in actuality this is a bubble sort, but hardcoded for the worst-case scenario to avoid "ifs". It still uses conditional swaps tough.
__device__ __host__ void sortArray(float &fv1, float &fv2, float &fv3, float &vx1, float &vy1, float &vx2, float &vy2, float &vx3, float &vy3)
{
        swapArray(fv2,vx2,vy2,fv3,vx3,vy3);
        swapArray(fv1,vx1,vy1,fv2,vx2,vy2);
        swapArray(fv2,vx2,vy2,fv3,vx3,vy3);
}
// Conditional swap of points for the sortArray function.
__device__ __host__ void swapArray(float &a1,float &a2, float &a3,float &b1,float &b2,float &b3)
{
        float tmp;
        if (a1 > b1)
        {
            tmp=b1; b1=a1; a1=tmp;
            tmp=b2; b2=a2; a2=tmp;
            tmp=b3; b3=a3; a3=tmp;
        }
}

// this method converts 17 input points (in A) to 5 coefficients of the polynomial that fits into these points
// note: this is my original research result. The code has been semi-machine-generated and then hand optimised
// details are described in a separate report.
__device__ __host__ void polyfit17x5(double output[5], double A[34])
{

// NOTE: These constants below are true constants, but #defines are not best for them because variables like R236 can be used elsewhere
// TODO: Rename generic names like R236 to routine-specific names to replace them with defined constants


double R342 = 17.;
double R193 = -1.;

double R34 = A[0];
double R35 = A[9];
double R36 = A[10];
double R37 = A[11];
double R38 = A[12];
double R39 = A[13];
double R40 = A[14];
double R41 = A[15];
double R42 = A[16];
double R43 = A[1];
double R44 = A[2];
double R45 = A[3];
double R46 = A[4];
double R47 = A[5];
double R48 = A[6];
double R49 = A[7];
double R50 = A[8];
double R51 = Square( A[0]);
double R52 = Square( A[9]);
double R53 = Square( A[10]);
double R54 = Square( A[11]);
double R55 = Square( A[12]);
double R56 = Square( A[13]);
double R57 = Square( A[14]);
double R58 = Square( A[15]);
double R59 = Square( A[16]);
double R60 = Square( A[1]);
double R61 = Square( A[2]);
double R62 = Square( A[3]);
double R63 = Square( A[4]);
double R64 = Square( A[5]);
double R65 = Square( A[6]);
double R66 = Square( A[7]);
double R67 = Square( A[8]);
double R68 = Square( R34);
double R69 = Square( R35);
double R70 = Square( R36);
double R71 = Square( R37);
double R72 = Square( R38);
double R73 = Square( R39);
double R74 = Square( R40);
double R75 = Square( R41);
double R76 = Square( R42);
double R77 = Square( R43);
double R78 = Square( R44);
double R79 = Square( R45);
double R80 = Square( R46);
double R81 = Square( R47);
double R82 = Square( R48);
double R83 = Square( R49);
double R84 = Square( R50);
double R85 = Power3( A[0]);
double R86 = Power3( A[9]);
double R87 = Power3( A[10]);
double R88 = Power3( A[11]);
double R89 = Power3( A[12]);
double R90 = Power3( A[13]);
double R91 = Power3( A[14]);
double R92 = Power3( A[15]);
double R93 = Power3( A[16]);
double R94 = Power3( A[1]);
double R95 = Power3( A[2]);
double R96 = Power3( A[3]);
double R97 = Power3( A[4]);
double R98 = Power3( A[5]);
double R99 = Power3( A[6]);
double R100 = Power3( A[7]);
double R101 = Power3( A[8]);
double R102 = A[0] * R34;
double R103 = A[9] * R35;
double R104 = A[10] * R36;
double R105 = A[11] * R37;
double R106 = A[12] * R38;
double R107 = A[13] * R39;
double R108 = A[14] * R40;
double R109 = A[15] * R41;
double R110 = A[16] * R42;
double R111 = A[1] * R43;
double R112 = A[2] * R44;
double R113 = A[3] * R45;
double R114 = A[4] * R46;
double R115 = A[5] * R47;
double R116 = A[6] * R48;
double R117 = A[7] * R49;
double R118 = A[8] * R50;
double R119 = R102 + R103 + R104 + R105 + R106 + R107 + R108 + R109 + R110 + R111 + R112 + R113 + R114 + R115 + R116 + R117 + R118;
double R120 = A[0] + A[9] + A[10] + A[11] + A[12] + A[13] + A[14] + A[15] + A[16] + A[1] + A[2] + A[3] + A[4] + A[5] + A[6] + A[7] + A[8];
double R121 = R85 + R86 + R87 + R88 + R89 + R90 + R91 + R92 + R93 + R94 + R95 + R96 + R97 + R98 + R99 + R100 + R101;
double R122 = R51 * R34;
double R123 = R52 * R35;
double R124 = R53 * R36;
double R125 = R54 * R37;
double R126 = R55 * R38;
double R127 = R56 * R39;
double R128 = R57 * R40;
double R129 = R58 * R41;
double R130 = R59 * R42;
double R131 = R60 * R43;
double R132 = R61 * R44;
double R133 = R62 * R45;
double R134 = R63 * R46;
double R135 = R64 * R47;
double R136 = R65 * R48;
double R137 = R66 * R49;
double R138 = R67 * R50;
double R139 = R122 + R123 + R124 + R125 + R126 + R127 + R128 + R129 + R130 + R131 + R132 + R133 + R134 + R135 + R136 + R137 + R138;
double R140 = R51 + R52 + R53 + R54 + R55 + R56 + R57 + R58 + R59 + R60 + R61 + R62 + R63 + R64 + R65 + R66 + R67;
double R141 = R85 * R34;
double R142 = R86 * R35;
double R143 = R87 * R36;
double R144 = R88 * R37;
double R145 = R89 * R38;
double R146 = R90 * R39;
double R147 = R91 * R40;
double R148 = R92 * R41;
double R149 = R93 * R42;
double R150 = R94 * R43;
double R151 = R95 * R44;
double R152 = R96 * R45;
double R153 = R97 * R46;
double R154 = R98 * R47;
double R155 = R99 * R48;
double R156 = R100 * R49;
double R157 = R101 * R50;
double R158 = R141 + R142 + R143 + R144 + R145 + R146 + R147 + R148 + R149 + R150 + R151 + R152 + R153 + R154 + R155 + R156 + R157;
double R159 = Power3( R34);
double R160 = Power3( R35);
double R161 = Power3( R36);
double R162 = Power3( R37);
double R163 = Power3( R38);
double R164 = Power3( R39);
double R165 = Power3( R40);
double R166 = Power3( R41);
double R167 = Power3( R42);
double R168 = Power3( R43);
double R169 = Power3( R44);
double R170 = Power3( R45);
double R171 = Power3( R46);
double R172 = Power3( R47);
double R173 = Power3( R48);
double R174 = Power3( R49);
double R175 = Power3( R50);
double R176 = Power4( A[0]);
double R177 = Power4( A[9]);
double R178 = Power4( A[10]);
double R179 = Power4( A[11]);
double R180 = Power4( A[12]);
double R181 = Power4( A[13]);
double R182 = Power4( A[14]);
double R183 = Power4( A[15]);
double R184 = Power4( A[16]);
double R185 = Power4( A[1]);
double R186 = Power4( A[2]);
double R187 = Power4( A[3]);
double R188 = Power4( A[4]);
double R189 = Power4( A[5]);
double R190 = Power4( A[6]);
double R191 = Power4( A[7]);
double R192 = Power4( A[8]);
double R194 = R193 * R140 * R119;
double R195 = R120 * R139;
double R196 = R194 + R195;
double R197 = R51 * R68;
double R198 = R52 * R69;
double R199 = R53 * R70;
double R200 = R54 * R71;
double R201 = R55 * R72;
double R202 = R56 * R73;
double R203 = R57 * R74;
double R204 = R58 * R75;
double R205 = R59 * R76;
double R206 = R60 * R77;
double R207 = R61 * R78;
double R208 = R62 * R79;
double R209 = R63 * R80;
double R210 = R64 * R81;
double R211 = R65 * R82;
double R212 = R66 * R83;
double R213 = R67 * R84;
double R214 = R197 + R198 + R199 + R200 + R201 + R202 + R203 + R204 + R205 + R206 + R207 + R208 + R209 + R210 + R211 + R212 + R213;
double R215 = A[0] * R68;
double R216 = A[9] * R69;
double R217 = A[10] * R70;
double R218 = A[11] * R71;
double R219 = A[12] * R72;
double R220 = A[13] * R73;
double R221 = A[14] * R74;
double R222 = A[15] * R75;
double R223 = A[16] * R76;
double R224 = A[1] * R77;
double R225 = A[2] * R78;
double R226 = A[3] * R79;
double R227 = A[4] * R80;
double R228 = A[5] * R81;
double R229 = A[6] * R82;
double R230 = A[7] * R83;
double R231 = A[8] * R84;
double R232 = R215 + R216 + R217 + R218 + R219 + R220 + R221 + R222 + R223 + R224 + R225 + R226 + R227 + R228 + R229 + R230 + R231;
double R233 = R176 + R177 + R178 + R179 + R180 + R181 + R182 + R183 + R184 + R185 + R186 + R187 + R188 + R189 + R190 + R191 + R192;
double R234 = R176 * R34;
double R235 = R177 * R35;
double R236 = R178 * R36;
double R237 = R179 * R37;
double R238 = R180 * R38;
double R239 = R181 * R39;
double R240 = R182 * R40;
double R241 = R183 * R41;
double R242 = R184 * R42;
double R243 = R185 * R43;
double R244 = R186 * R44;
double R245 = R187 * R45;
double R246 = R188 * R46;
double R247 = R189 * R47;
double R248 = R190 * R48;
double R249 = R191 * R49;
double R250 = R192 * R50;
double R251 = R234 + R235 + R236 + R237 + R238 + R239 + R240 + R241 + R242 + R243 + R244 + R245 + R246 + R247 + R248 + R249 + R250;
double R252 = R176 * R68;
double R253 = R177 * R69;
double R254 = R178 * R70;
double R255 = R179 * R71;
double R256 = R180 * R72;
double R257 = R181 * R73;
double R258 = R182 * R74;
double R259 = R183 * R75;
double R260 = R184 * R76;
double R261 = R185 * R77;
double R262 = R186 * R78;
double R263 = R187 * R79;
double R264 = R188 * R80;
double R265 = R189 * R81;
double R266 = R190 * R82;
double R267 = R191 * R83;
double R268 = R192 * R84;
double R269 = R252 + R253 + R254 + R255 + R256 + R257 + R258 + R259 + R260 + R261 + R262 + R263 + R264 + R265 + R266 + R267 + R268;
double R270 = R193 * R121 * R119;
double R271 = R120 * R158;
double R272 = R270 + R271;
double R273 = R85 * R68;
double R274 = R86 * R69;
double R275 = R87 * R70;
double R276 = R88 * R71;
double R277 = R89 * R72;
double R278 = R90 * R73;
double R279 = R91 * R74;
double R280 = R92 * R75;
double R281 = R93 * R76;
double R282 = R94 * R77;
double R283 = R95 * R78;
double R284 = R96 * R79;
double R285 = R97 * R80;
double R286 = R98 * R81;
double R287 = R99 * R82;
double R288 = R100 * R83;
double R289 = R101 * R84;
double R290 = R273 + R274 + R275 + R276 + R277 + R278 + R279 + R280 + R281 + R282 + R283 + R284 + R285 + R286 + R287 + R288 + R289;
double R291 = R193 * R233 * R119;
double R292 = R120 * R251;
double R293 = R291 + R292;
double R294 = R193 * R121 * R139;
double R295 = R140 * R158;
double R296 = R294 + R295;
double R297 = R193 * R233 * R139;
double R298 = R140 * R251;
double R299 = R297 + R298;
double R300 = R193 * R233 * R158;
double R301 = R121 * R251;
double R302 = R300 + R301;
double R303 = Power4( R34);
double R304 = R85 * R159;
double R305 = R86 * R160;
double R306 = R87 * R161;
double R307 = R88 * R162;
double R308 = R89 * R163;
double R309 = R90 * R164;
double R310 = R91 * R165;
double R311 = R92 * R166;
double R312 = R93 * R167;
double R313 = R94 * R168;
double R314 = R95 * R169;
double R315 = R96 * R170;
double R316 = R97 * R171;
double R317 = R98 * R172;
double R318 = R99 * R173;
double R319 = R100 * R174;
double R320 = R101 * R175;
double R321 = R304 + R305 + R306 + R307 + R308 + R309 + R310 + R311 + R312 + R313 + R314 + R315 + R316 + R317 + R318 + R319 + R320;
double R322 = R34 + R35 + R36 + R37 + R38 + R39 + R40 + R41 + R42 + R43 + R44 + R45 + R46 + R47 + R48 + R49 + R50;
double R323 = R51 * R159;
double R324 = R52 * R160;
double R325 = R53 * R161;
double R326 = R54 * R162;
double R327 = R55 * R163;
double R328 = R56 * R164;
double R329 = R57 * R165;
double R330 = R58 * R166;
double R331 = R59 * R167;
double R332 = R60 * R168;
double R333 = R61 * R169;
double R334 = R62 * R170;
double R335 = R63 * R171;
double R336 = R64 * R172;
double R337 = R65 * R173;
double R338 = R66 * R174;
double R339 = R67 * R175;
double R340 = R323 + R324 + R325 + R326 + R327 + R328 + R329 + R330 + R331 + R332 + R333 + R334 + R335 + R336 + R337 + R338 + R339;
double R341 = R193 * R120 * R322;
double R343 = R342 * R119;
double R344 = R341 + R343;
double R345 = R68 + R69 + R70 + R71 + R72 + R73 + R74 + R75 + R76 + R77 + R78 + R79 + R80 + R81 + R82 + R83 + R84;
double R346 = A[0] * R159;
double R347 = A[9] * R160;
double R348 = A[10] * R161;
double R349 = A[11] * R162;
double R350 = A[12] * R163;
double R351 = A[13] * R164;
double R352 = A[14] * R165;
double R353 = A[15] * R166;
double R354 = A[16] * R167;
double R355 = A[1] * R168;
double R356 = A[2] * R169;
double R357 = A[3] * R170;
double R358 = A[4] * R171;
double R359 = A[5] * R172;
double R360 = A[6] * R173;
double R361 = A[7] * R174;
double R362 = A[8] * R175;
double R363 = R346 + R347 + R348 + R349 + R350 + R351 + R352 + R353 + R354 + R355 + R356 + R357 + R358 + R359 + R360 + R361 + R362;
double R364 = R193 * R140 * R322;
double R365 = R342 * R139;
double R366 = R364 + R365;
double R367 = R193 * R121 * R322;
double R368 = R342 * R158;
double R369 = R367 + R368;
double R370 = R290 * R196;
double R371 = R193 * R214 * R272;
double R372 = R232 * R296;
double R373 = R370 + R371 + R372;
double R374 = Power4( R35);
double R375 = Power4( R36);
double R376 = Power4( R37);
double R377 = Power4( R38);
double R378 = Power4( R39);
double R379 = Power4( R40);
double R380 = Power4( R41);
double R381 = Power4( R42);
double R382 = Power4( R43);
double R383 = Power4( R44);
double R384 = Power4( R45);
double R385 = Power4( R46);
double R386 = Power4( R47);
double R387 = Power4( R48);
double R388 = Power4( R49);
double R389 = Power4( R50);
double R390 = R176 * R159;
double R391 = R177 * R160;
double R392 = R178 * R161;
double R393 = R179 * R162;
double R394 = R180 * R163;
double R395 = R181 * R164;
double R396 = R182 * R165;
double R397 = R183 * R166;
double R398 = R184 * R167;
double R399 = R185 * R168;
double R400 = R186 * R169;
double R401 = R187 * R170;
double R402 = R188 * R171;
double R403 = R189 * R172;
double R404 = R190 * R173;
double R405 = R191 * R174;
double R406 = R192 * R175;
double R407 = R390 + R391 + R392 + R393 + R394 + R395 + R396 + R397 + R398 + R399 + R400 + R401 + R402 + R403 + R404 + R405 + R406;
double R408 = R214 * R344;
double R409 = R193 * R232 * R366;
double R410 = R345 * R196;
double R411 = R408 + R409 + R410;
double R412 = R193 * R233 * R322;
double R413 = R342 * R251;
double R414 = R412 + R413;
double R415 = R159 + R160 + R161 + R162 + R163 + R164 + R165 + R166 + R167 + R168 + R169 + R170 + R171 + R172 + R173 + R174 + R175;
double R416 = R269 * R196;
double R417 = R193 * R214 * R293;
double R418 = R232 * R299;
double R419 = R416 + R417 + R418;
double R420 = R290 * R344;
double R421 = R193 * R232 * R369;
double R422 = R345 * R272;
double R423 = R420 + R421 + R422;
double R424 = R269 * R344;
double R425 = R193 * R232 * R414;
double R426 = R345 * R293;
double R427 = R424 + R425 + R426;
double R428 = R269 * R272;
double R429 = R193 * R290 * R293;
double R430 = R232 * R302;
double R431 = R428 + R429 + R430;
double R432 = R290 * R366;
double R433 = R193 * R214 * R369;
double R434 = R345 * R296;
double R435 = R432 + R433 + R434;
double R436 = R269 * R366;
double R437 = R193 * R214 * R414;
double R438 = R345 * R299;
double R439 = R436 + R437 + R438;
double R440 = R269 * R369;
double R441 = R193 * R290 * R414;
double R442 = R345 * R302;
double R443 = R440 + R441 + R442;
double R444 = R269 * R296;
double R445 = R193 * R290 * R299;
double R446 = R214 * R302;
double R447 = R444 + R445 + R446;
double R448 = R407 * R373;
double R449 = R193 * R321 * R419;
double R450 = R340 * R431;
double R451 = R193 * R363 * R447;
double R452 = R448 + R449 + R450 + R451;
double R453 = R176 * R303;
double R454 = R177 * R374;
double R455 = R178 * R375;
double R456 = R179 * R376;
double R457 = R180 * R377;
double R458 = R181 * R378;
double R459 = R182 * R379;
double R460 = R183 * R380;
double R461 = R184 * R381;
double R462 = R185 * R382;
double R463 = R186 * R383;
double R464 = R187 * R384;
double R465 = R188 * R385;
double R466 = R189 * R386;
double R467 = R190 * R387;
double R468 = R191 * R388;
double R469 = R192 * R389;
double R470 = R453 + R454 + R455 + R456 + R457 + R458 + R459 + R460 + R461 + R462 + R463 + R464 + R465 + R466 + R467 + R468 + R469;
double R471 = R85 * R303;
double R472 = R86 * R374;
double R473 = R87 * R375;
double R474 = R88 * R376;
double R475 = R89 * R377;
double R476 = R90 * R378;
double R477 = R91 * R379;
double R478 = R92 * R380;
double R479 = R93 * R381;
double R480 = R94 * R382;
double R481 = R95 * R383;
double R482 = R96 * R384;
double R483 = R97 * R385;
double R484 = R98 * R386;
double R485 = R99 * R387;
double R486 = R100 * R388;
double R487 = R101 * R389;
double R488 = R471 + R472 + R473 + R474 + R475 + R476 + R477 + R478 + R479 + R480 + R481 + R482 + R483 + R484 + R485 + R486 + R487;
double R489 = R51 * R303;
double R490 = R52 * R374;
double R491 = R53 * R375;
double R492 = R54 * R376;
double R493 = R55 * R377;
double R494 = R56 * R378;
double R495 = R57 * R379;
double R496 = R58 * R380;
double R497 = R59 * R381;
double R498 = R60 * R382;
double R499 = R61 * R383;
double R500 = R62 * R384;
double R501 = R63 * R385;
double R502 = R64 * R386;
double R503 = R65 * R387;
double R504 = R66 * R388;
double R505 = R67 * R389;
double R506 = R489 + R490 + R491 + R492 + R493 + R494 + R495 + R496 + R497 + R498 + R499 + R500 + R501 + R502 + R503 + R504 + R505;
double R507 = A[0] * R303;
double R508 = A[9] * R374;
double R509 = A[10] * R375;
double R510 = A[11] * R376;
double R511 = A[12] * R377;
double R512 = A[13] * R378;
double R513 = A[14] * R379;
double R514 = A[15] * R380;
double R515 = A[16] * R381;
double R516 = A[1] * R382;
double R517 = A[2] * R383;
double R518 = A[3] * R384;
double R519 = A[4] * R385;
double R520 = A[5] * R386;
double R521 = A[6] * R387;
double R522 = A[7] * R388;
double R523 = A[8] * R389;
double R524 = R507 + R508 + R509 + R510 + R511 + R512 + R513 + R514 + R515 + R516 + R517 + R518 + R519 + R520 + R521 + R522 + R523;
double R525 = R321 * R411;
double R526 = R193 * R340 * R423;
double R527 = R363 * R435;
double R528 = R193 * R415 * R373;
double R529 = R525 + R526 + R527 + R528;
double R530 = R470 * R529;
double R531 = R407 * R411;
double R532 = R193 * R340 * R427;
double R533 = R363 * R439;
double R534 = R193 * R415 * R419;
double R535 = R531 + R532 + R533 + R534;
double R536 = R193 * R488 * R535;
double R537 = R407 * R423;
double R538 = R193 * R321 * R427;
double R539 = R363 * R443;
double R540 = R193 * R415 * R431;
double R541 = R537 + R538 + R539 + R540;
double R542 = R506 * R541;
double R543 = R407 * R435;
double R544 = R193 * R321 * R439;
double R545 = R340 * R443;
double R546 = R193 * R415 * R447;
double R547 = R543 + R544 + R545 + R546;
double R548 = R193 * R524 * R547;
double R549 = R303 + R374 + R375 + R376 + R377 + R378 + R379 + R380 + R381 + R382 + R383 + R384 + R385 + R386 + R387 + R388 + R389;
double R550 = R549 * R452;
double R551 = R530 + R536 + R542 + R548 + R550;
double R552 = Reciprocal( R551);
double R553 = R193 * R140 * R232;
double R554 = R120 * R214;
double R555 = R553 + R554;
double R556 = R193 * R121 * R232;
double R557 = R120 * R290;
double R558 = R556 + R557;
double R559 = R193 * R233 * R232;
double R560 = R120 * R269;
double R561 = R559 + R560;
double R562 = R193 * R121 * R214;
double R563 = R140 * R290;
double R564 = R562 + R563;
double R565 = R193 * R233 * R214;
double R566 = R140 * R269;
double R567 = R565 + R566;
double R568 = R193 * R233 * R290;
double R569 = R121 * R269;
double R570 = R568 + R569;
double R571 = R193 * R139 * R232;
double R572 = R119 * R214;
double R573 = R571 + R572;
double R574 = R193 * R158 * R232;
double R575 = R119 * R290;
double R576 = R574 + R575;
double R577 = R193 * R251 * R232;
double R578 = R119 * R269;
double R579 = R577 + R578;
double R580 = R193 * R158 * R214;
double R581 = R139 * R290;
double R582 = R580 + R581;
double R583 = R193 * R251 * R214;
double R584 = R139 * R269;
double R585 = R583 + R584;
double R586 = R193 * R251 * R290;
double R587 = R158 * R269;
double R588 = R586 + R587;
double R589 = R193 * R470 * R373;
double R590 = R488 * R419;
double R591 = R193 * R506 * R431;
double R592 = R524 * R447;
double R593 = R589 + R590 + R591 + R592;
double R594 = R321 * R196;
double R595 = R193 * R340 * R272;
double R596 = R363 * R296;
double R597 = R594 + R595 + R596;
double R598 = R470 * R597;
double R599 = R407 * R196;
double R600 = R193 * R340 * R293;
double R601 = R363 * R299;
double R602 = R599 + R600 + R601;
double R603 = R193 * R488 * R602;
double R604 = R407 * R272;
double R605 = R193 * R321 * R293;
double R606 = R363 * R302;
double R607 = R604 + R605 + R606;
double R608 = R506 * R607;
double R609 = R407 * R296;
double R610 = R193 * R321 * R299;
double R611 = R340 * R302;
double R612 = R609 + R610 + R611;
double R613 = R193 * R524 * R612;
double R614 = R598 + R603 + R608 + R613;
double R615 = R321 * R555;
double R616 = R193 * R340 * R558;
double R617 = R363 * R564;
double R618 = R615 + R616 + R617;
double R619 = R193 * R470 * R618;
double R620 = R407 * R555;
double R621 = R193 * R340 * R561;
double R622 = R363 * R567;
double R623 = R620 + R621 + R622;
double R624 = R488 * R623;
double R625 = R407 * R558;
double R626 = R193 * R321 * R561;
double R627 = R363 * R570;
double R628 = R625 + R626 + R627;
double R629 = R193 * R506 * R628;
double R630 = R407 * R564;
double R631 = R193 * R321 * R567;
double R632 = R340 * R570;
double R633 = R630 + R631 + R632;
double R634 = R524 * R633;
double R635 = R619 + R624 + R629 + R634;
double R636 = R321 * R573;
double R637 = R193 * R340 * R576;
double R638 = R363 * R582;
double R639 = R636 + R637 + R638;
double R640 = R470 * R639;
double R641 = R407 * R573;
double R642 = R193 * R340 * R579;
double R643 = R363 * R585;
double R644 = R641 + R642 + R643;
double R645 = R193 * R488 * R644;
double R646 = R407 * R576;
double R647 = R193 * R321 * R579;
double R648 = R363 * R588;
double R649 = R646 + R647 + R648;
double R650 = R506 * R649;
double R651 = R407 * R582;
double R652 = R193 * R321 * R585;
double R653 = R340 * R588;
double R654 = R651 + R652 + R653;
double R655 = R193 * R524 * R654;
double R656 = R640 + R645 + R650 + R655;
double R657 = R656 * R552;
double R658 = R193 * R140 * R345;
double R659 = R342 * R214;
double R660 = R658 + R659;
double R661 = R193 * R121 * R345;
double R662 = R342 * R290;
double R663 = R661 + R662;
double R664 = R193 * R233 * R345;
double R665 = R342 * R269;
double R666 = R664 + R665;
double R667 = R193 * R139 * R345;
double R668 = R322 * R214;
double R669 = R667 + R668;
double R670 = R193 * R158 * R345;
double R671 = R322 * R290;
double R672 = R670 + R671;
double R673 = R193 * R251 * R345;
double R674 = R322 * R269;
double R675 = R673 + R674;
double R676 = R193 * R407 * R435;
double R677 = R321 * R439;
double R678 = R193 * R340 * R443;
double R679 = R415 * R447;
double R680 = R676 + R677 + R678 + R679;
double R681 = R470 * R435;
double R682 = R193 * R488 * R439;
double R683 = R506 * R443;
double R684 = R193 * R549 * R447;
double R685 = R681 + R682 + R683 + R684;
double R686 = R321 * R366;
double R687 = R193 * R340 * R369;
double R688 = R415 * R296;
double R689 = R686 + R687 + R688;
double R690 = R193 * R470 * R689;
double R691 = R407 * R366;
double R692 = R193 * R340 * R414;
double R693 = R415 * R299;
double R694 = R691 + R692 + R693;
double R695 = R488 * R694;
double R696 = R407 * R369;
double R697 = R193 * R321 * R414;
double R698 = R415 * R302;
double R699 = R696 + R697 + R698;
double R700 = R193 * R506 * R699;
double R701 = R549 * R612;
double R702 = R690 + R695 + R700 + R701;
double R703 = R321 * R660;
double R704 = R193 * R340 * R663;
double R705 = R415 * R564;
double R706 = R703 + R704 + R705;
double R707 = R470 * R706;
double R708 = R407 * R660;
double R709 = R193 * R340 * R666;
double R710 = R415 * R567;
double R711 = R708 + R709 + R710;
double R712 = R193 * R488 * R711;
double R713 = R407 * R663;
double R714 = R193 * R321 * R666;
double R715 = R415 * R570;
double R716 = R713 + R714 + R715;
double R717 = R506 * R716;
double R718 = R193 * R549 * R633;
double R719 = R707 + R712 + R717 + R718;
double R720 = R321 * R669;
double R721 = R193 * R340 * R672;
double R722 = R415 * R582;
double R723 = R720 + R721 + R722;
double R724 = R193 * R470 * R723;
double R725 = R407 * R669;
double R726 = R193 * R340 * R675;
double R727 = R415 * R585;
double R728 = R725 + R726 + R727;
double R729 = R488 * R728;
double R730 = R407 * R672;
double R731 = R193 * R321 * R675;
double R732 = R415 * R588;
double R733 = R730 + R731 + R732;
double R734 = R193 * R506 * R733;
double R735 = R549 * R654;
double R736 = R724 + R729 + R734 + R735;
double R737 = R736 * R552;
double R738 = R193 * R120 * R345;
double R739 = R342 * R232;
double R740 = R738 + R739;
double R741 = R193 * R119 * R345;
double R742 = R322 * R232;
double R743 = R741 + R742;
double R744 = R193 * R470 * R423;
double R745 = R488 * R427;
double R746 = R193 * R524 * R443;
double R747 = R549 * R431;
double R748 = R744 + R745 + R746 + R747;
double R749 = R321 * R344;
double R750 = R193 * R363 * R369;
double R751 = R415 * R272;
double R752 = R749 + R750 + R751;
double R753 = R470 * R752;
double R754 = R407 * R344;
double R755 = R193 * R363 * R414;
double R756 = R415 * R293;
double R757 = R754 + R755 + R756;
double R758 = R193 * R488 * R757;
double R759 = R524 * R699;
double R760 = R193 * R549 * R607;
double R761 = R753 + R758 + R759 + R760;
double R762 = R321 * R740;
double R763 = R193 * R363 * R663;
double R764 = R415 * R558;
double R765 = R762 + R763 + R764;
double R766 = R193 * R470 * R765;
double R767 = R407 * R740;
double R768 = R193 * R363 * R666;
double R769 = R415 * R561;
double R770 = R767 + R768 + R769;
double R771 = R488 * R770;
double R772 = R193 * R524 * R716;
double R773 = R549 * R628;
double R774 = R766 + R771 + R772 + R773;
double R775 = R321 * R743;
double R776 = R193 * R363 * R672;
double R777 = R415 * R576;
double R778 = R775 + R776 + R777;
double R779 = R470 * R778;
double R780 = R407 * R743;
double R781 = R193 * R363 * R675;
double R782 = R415 * R579;
double R783 = R780 + R781 + R782;
double R784 = R193 * R488 * R783;
double R785 = R524 * R733;
double R786 = R193 * R549 * R649;
double R787 = R779 + R784 + R785 + R786;
double R788 = R787 * R552;
double R789 = R193 * R407 * R411;
double R790 = R340 * R427;
double R791 = R193 * R363 * R439;
double R792 = R415 * R419;
double R793 = R789 + R790 + R791 + R792;
double R794 = R470 * R411;
double R795 = R193 * R506 * R427;
double R796 = R524 * R439;
double R797 = R193 * R549 * R419;
double R798 = R794 + R795 + R796 + R797;
double R799 = R340 * R344;
double R800 = R193 * R363 * R366;
double R801 = R415 * R196;
double R802 = R799 + R800 + R801;
double R803 = R193 * R470 * R802;
double R804 = R506 * R757;
double R805 = R193 * R524 * R694;
double R806 = R549 * R602;
double R807 = R803 + R804 + R805 + R806;
double R808 = R340 * R740;
double R809 = R193 * R363 * R660;
double R810 = R415 * R555;
double R811 = R808 + R809 + R810;
double R812 = R470 * R811;
double R813 = R193 * R506 * R770;
double R814 = R524 * R711;
double R815 = R193 * R549 * R623;
double R816 = R812 + R813 + R814 + R815;
double R817 = R340 * R743;
double R818 = R193 * R363 * R669;
double R819 = R415 * R573;
double R820 = R817 + R818 + R819;
double R821 = R193 * R470 * R820;
double R822 = R506 * R783;
double R823 = R193 * R524 * R728;
double R824 = R549 * R644;
double R825 = R821 + R822 + R823 + R824;
double R826 = R825 * R552;
double R827 = R193 * R488 * R411;
double R828 = R506 * R423;
double R829 = R193 * R524 * R435;
double R830 = R549 * R373;
double R831 = R827 + R828 + R829 + R830;
double R832 = R488 * R802;
double R833 = R193 * R506 * R752;
double R834 = R524 * R689;
double R835 = R193 * R549 * R597;
double R836 = R832 + R833 + R834 + R835;
double R837 = R193 * R488 * R811;
double R838 = R506 * R765;
double R839 = R193 * R524 * R706;
double R840 = R549 * R618;
double R841 = R837 + R838 + R839 + R840;
double R842 = R488 * R820;
double R843 = R193 * R506 * R778;
double R844 = R524 * R723;
double R845 = R193 * R549 * R639;
double R846 = R842 + R843 + R844 + R845;
double R847 = R846 * R552;
double R848 = R303 * R452 * R552;
double R849 = R159 * R593 * R552;
double R850 = R68 * R614 * R552;
double R851 = R34 * R635 * R552;
R848 = R848 + R849 + R850 + R851 + R657;
R849 = A[17] * R848;
R848 = R374 * R452 * R552;
R850 = R160 * R593 * R552;
R851 = R69 * R614 * R552;
double R852 = R35 * R635 * R552;
R848 = R848 + R850 + R851 + R852 + R657;
R850 = A[26] * R848;
R848 = R375 * R452 * R552;
R851 = R161 * R593 * R552;
R852 = R70 * R614 * R552;
double R853 = R36 * R635 * R552;
R848 = R848 + R851 + R852 + R853 + R657;
R851 = A[27] * R848;
R848 = R376 * R452 * R552;
R852 = R162 * R593 * R552;
R853 = R71 * R614 * R552;
double R854 = R37 * R635 * R552;
R848 = R848 + R852 + R853 + R854 + R657;
R852 = A[28] * R848;
R848 = R377 * R452 * R552;
R853 = R163 * R593 * R552;
R854 = R72 * R614 * R552;
double R855 = R38 * R635 * R552;
R848 = R848 + R853 + R854 + R855 + R657;
R853 = A[29] * R848;
R848 = R378 * R452 * R552;
R854 = R164 * R593 * R552;
R855 = R73 * R614 * R552;
double R856 = R39 * R635 * R552;
R848 = R848 + R854 + R855 + R856 + R657;
R854 = A[30] * R848;
R848 = R379 * R452 * R552;
R855 = R165 * R593 * R552;
R856 = R74 * R614 * R552;
double R857 = R40 * R635 * R552;
R848 = R848 + R855 + R856 + R857 + R657;
R855 = A[31] * R848;
R848 = R380 * R452 * R552;
R856 = R166 * R593 * R552;
R857 = R75 * R614 * R552;
double R858 = R41 * R635 * R552;
R848 = R848 + R856 + R857 + R858 + R657;
R856 = A[32] * R848;
R848 = R381 * R452 * R552;
R857 = R167 * R593 * R552;
R858 = R76 * R614 * R552;
double R859 = R42 * R635 * R552;
R848 = R848 + R857 + R858 + R859 + R657;
R857 = A[33] * R848;
R848 = R382 * R452 * R552;
R858 = R168 * R593 * R552;
R859 = R77 * R614 * R552;
double R860 = R43 * R635 * R552;
R848 = R848 + R858 + R859 + R860 + R657;
R858 = A[18] * R848;
R848 = R383 * R452 * R552;
R859 = R169 * R593 * R552;
R860 = R78 * R614 * R552;
double R861 = R44 * R635 * R552;
R848 = R848 + R859 + R860 + R861 + R657;
R859 = A[19] * R848;
R848 = R384 * R452 * R552;
R860 = R170 * R593 * R552;
R861 = R79 * R614 * R552;
double R862 = R45 * R635 * R552;
R848 = R848 + R860 + R861 + R862 + R657;
R860 = A[20] * R848;
R848 = R385 * R452 * R552;
R861 = R171 * R593 * R552;
R862 = R80 * R614 * R552;
double R863 = R46 * R635 * R552;
R848 = R848 + R861 + R862 + R863 + R657;
R861 = A[21] * R848;
R848 = R386 * R452 * R552;
R862 = R172 * R593 * R552;
R863 = R81 * R614 * R552;
double R864 = R47 * R635 * R552;
R848 = R848 + R862 + R863 + R864 + R657;
R862 = A[22] * R848;
R848 = R387 * R452 * R552;
R863 = R173 * R593 * R552;
R864 = R82 * R614 * R552;
double R865 = R48 * R635 * R552;
R848 = R848 + R863 + R864 + R865 + R657;
R863 = A[23] * R848;
R848 = R388 * R452 * R552;
R864 = R174 * R593 * R552;
R865 = R83 * R614 * R552;
double R866 = R49 * R635 * R552;
R848 = R848 + R864 + R865 + R866 + R657;
R864 = A[24] * R848;
R848 = R389 * R452 * R552;
R865 = R175 * R593 * R552;
R866 = R84 * R614 * R552;
double R867 = R50 * R635 * R552;
R848 = R848 + R865 + R866 + R867 + R657;
R865 = A[25] * R848;
R849 = R849 + R850 + R851 + R852 + R853 + R854 + R855 + R856 + R857 + R858 + R859 + R860 + R861 + R862 + R863 + R864 + R865;
R850 = R303 * R680 * R552;
R851 = R159 * R685 * R552;
R852 = R68 * R702 * R552;
R853 = R34 * R719 * R552;
R850 = R850 + R851 + R852 + R853 + R737;
R851 = A[17] * R850;
R850 = R374 * R680 * R552;
R852 = R160 * R685 * R552;
R853 = R69 * R702 * R552;
R854 = R35 * R719 * R552;
R850 = R850 + R852 + R853 + R854 + R737;
R852 = A[26] * R850;
R850 = R375 * R680 * R552;
R853 = R161 * R685 * R552;
R854 = R70 * R702 * R552;
R855 = R36 * R719 * R552;
R850 = R850 + R853 + R854 + R855 + R737;
R853 = A[27] * R850;
R850 = R376 * R680 * R552;
R854 = R162 * R685 * R552;
R855 = R71 * R702 * R552;
R856 = R37 * R719 * R552;
R850 = R850 + R854 + R855 + R856 + R737;
R854 = A[28] * R850;
R850 = R377 * R680 * R552;
R855 = R163 * R685 * R552;
R856 = R72 * R702 * R552;
R857 = R38 * R719 * R552;
R850 = R850 + R855 + R856 + R857 + R737;
R855 = A[29] * R850;
R850 = R378 * R680 * R552;
R856 = R164 * R685 * R552;
R857 = R73 * R702 * R552;
R858 = R39 * R719 * R552;
R850 = R850 + R856 + R857 + R858 + R737;
R856 = A[30] * R850;
R850 = R379 * R680 * R552;
R857 = R165 * R685 * R552;
R858 = R74 * R702 * R552;
R859 = R40 * R719 * R552;
R850 = R850 + R857 + R858 + R859 + R737;
R857 = A[31] * R850;
R850 = R380 * R680 * R552;
R858 = R166 * R685 * R552;
R859 = R75 * R702 * R552;
R860 = R41 * R719 * R552;
R850 = R850 + R858 + R859 + R860 + R737;
R858 = A[32] * R850;
R850 = R381 * R680 * R552;
R859 = R167 * R685 * R552;
R860 = R76 * R702 * R552;
R861 = R42 * R719 * R552;
R850 = R850 + R859 + R860 + R861 + R737;
R859 = A[33] * R850;
R850 = R382 * R680 * R552;
R860 = R168 * R685 * R552;
R861 = R77 * R702 * R552;
R862 = R43 * R719 * R552;
R850 = R850 + R860 + R861 + R862 + R737;
R860 = A[18] * R850;
R850 = R383 * R680 * R552;
R861 = R169 * R685 * R552;
R862 = R78 * R702 * R552;
R863 = R44 * R719 * R552;
R850 = R850 + R861 + R862 + R863 + R737;
R861 = A[19] * R850;
R850 = R384 * R680 * R552;
R862 = R170 * R685 * R552;
R863 = R79 * R702 * R552;
R864 = R45 * R719 * R552;
R850 = R850 + R862 + R863 + R864 + R737;
R862 = A[20] * R850;
R850 = R385 * R680 * R552;
R863 = R171 * R685 * R552;
R864 = R80 * R702 * R552;
R865 = R46 * R719 * R552;
R850 = R850 + R863 + R864 + R865 + R737;
R863 = A[21] * R850;
R850 = R386 * R680 * R552;
R864 = R172 * R685 * R552;
R865 = R81 * R702 * R552;
R848 = R47 * R719 * R552;
R850 = R850 + R864 + R865 + R848 + R737;
R864 = A[22] * R850;
R850 = R387 * R680 * R552;
R865 = R173 * R685 * R552;
R848 = R82 * R702 * R552;
R866 = R48 * R719 * R552;
R850 = R850 + R865 + R848 + R866 + R737;
R865 = A[23] * R850;
R850 = R388 * R680 * R552;
R848 = R174 * R685 * R552;
R866 = R83 * R702 * R552;
R867 = R49 * R719 * R552;
R850 = R850 + R848 + R866 + R867 + R737;
R848 = A[24] * R850;
R850 = R389 * R680 * R552;
R866 = R175 * R685 * R552;
R867 = R84 * R702 * R552;
double R868 = R50 * R719 * R552;
R850 = R850 + R866 + R867 + R868 + R737;
R866 = A[25] * R850;
R851 = R851 + R852 + R853 + R854 + R855 + R856 + R857 + R858 + R859 + R860 + R861 + R862 + R863 + R864 + R865 + R848 + R866;
R852 = R303 * R541 * R552;
R853 = R159 * R748 * R552;
R854 = R68 * R761 * R552;
R855 = R34 * R774 * R552;
R852 = R852 + R853 + R854 + R855 + R788;
R853 = A[17] * R852;
R852 = R374 * R541 * R552;
R854 = R160 * R748 * R552;
R855 = R69 * R761 * R552;
R856 = R35 * R774 * R552;
R852 = R852 + R854 + R855 + R856 + R788;
R854 = A[26] * R852;
R852 = R375 * R541 * R552;
R855 = R161 * R748 * R552;
R856 = R70 * R761 * R552;
R857 = R36 * R774 * R552;
R852 = R852 + R855 + R856 + R857 + R788;
R855 = A[27] * R852;
R852 = R376 * R541 * R552;
R856 = R162 * R748 * R552;
R857 = R71 * R761 * R552;
R858 = R37 * R774 * R552;
R852 = R852 + R856 + R857 + R858 + R788;
R856 = A[28] * R852;
R852 = R377 * R541 * R552;
R857 = R163 * R748 * R552;
R858 = R72 * R761 * R552;
R859 = R38 * R774 * R552;
R852 = R852 + R857 + R858 + R859 + R788;
R857 = A[29] * R852;
R852 = R378 * R541 * R552;
R858 = R164 * R748 * R552;
R859 = R73 * R761 * R552;
R860 = R39 * R774 * R552;
R852 = R852 + R858 + R859 + R860 + R788;
R858 = A[30] * R852;
R852 = R379 * R541 * R552;
R859 = R165 * R748 * R552;
R860 = R74 * R761 * R552;
R861 = R40 * R774 * R552;
R852 = R852 + R859 + R860 + R861 + R788;
R859 = A[31] * R852;
R852 = R380 * R541 * R552;
R860 = R166 * R748 * R552;
R861 = R75 * R761 * R552;
R862 = R41 * R774 * R552;
R852 = R852 + R860 + R861 + R862 + R788;
R860 = A[32] * R852;
R852 = R381 * R541 * R552;
R861 = R167 * R748 * R552;
R862 = R76 * R761 * R552;
R863 = R42 * R774 * R552;
R852 = R852 + R861 + R862 + R863 + R788;
R861 = A[33] * R852;
R852 = R382 * R541 * R552;
R862 = R168 * R748 * R552;
R863 = R77 * R761 * R552;
R864 = R43 * R774 * R552;
R852 = R852 + R862 + R863 + R864 + R788;
R862 = A[18] * R852;
R852 = R383 * R541 * R552;
R863 = R169 * R748 * R552;
R864 = R78 * R761 * R552;
R865 = R44 * R774 * R552;
R852 = R852 + R863 + R864 + R865 + R788;
R863 = A[19] * R852;
R852 = R384 * R541 * R552;
R864 = R170 * R748 * R552;
R865 = R79 * R761 * R552;
R848 = R45 * R774 * R552;
R852 = R852 + R864 + R865 + R848 + R788;
R864 = A[20] * R852;
R852 = R385 * R541 * R552;
R865 = R171 * R748 * R552;
R848 = R80 * R761 * R552;
R866 = R46 * R774 * R552;
R852 = R852 + R865 + R848 + R866 + R788;
R865 = A[21] * R852;
R852 = R386 * R541 * R552;
R848 = R172 * R748 * R552;
R866 = R81 * R761 * R552;
R850 = R47 * R774 * R552;
R852 = R852 + R848 + R866 + R850 + R788;
R848 = A[22] * R852;
R852 = R387 * R541 * R552;
R866 = R173 * R748 * R552;
R850 = R82 * R761 * R552;
R867 = R48 * R774 * R552;
R852 = R852 + R866 + R850 + R867 + R788;
R866 = A[23] * R852;
R852 = R388 * R541 * R552;
R850 = R174 * R748 * R552;
R867 = R83 * R761 * R552;
R868 = R49 * R774 * R552;
R852 = R852 + R850 + R867 + R868 + R788;
R850 = A[24] * R852;
R852 = R389 * R541 * R552;
R867 = R175 * R748 * R552;
R868 = R84 * R761 * R552;
double R869 = R50 * R774 * R552;
R852 = R852 + R867 + R868 + R869 + R788;
R867 = A[25] * R852;
R853 = R853 + R854 + R855 + R856 + R857 + R858 + R859 + R860 + R861 + R862 + R863 + R864 + R865 + R848 + R866 + R850 + R867;
R854 = R303 * R793 * R552;
R855 = R159 * R798 * R552;
R856 = R68 * R807 * R552;
R857 = R34 * R816 * R552;
R854 = R854 + R855 + R856 + R857 + R826;
R855 = A[17] * R854;
R854 = R374 * R793 * R552;
R856 = R160 * R798 * R552;
R857 = R69 * R807 * R552;
R858 = R35 * R816 * R552;
R854 = R854 + R856 + R857 + R858 + R826;
R856 = A[26] * R854;
R854 = R375 * R793 * R552;
R857 = R161 * R798 * R552;
R858 = R70 * R807 * R552;
R859 = R36 * R816 * R552;
R854 = R854 + R857 + R858 + R859 + R826;
R857 = A[27] * R854;
R854 = R376 * R793 * R552;
R858 = R162 * R798 * R552;
R859 = R71 * R807 * R552;
R860 = R37 * R816 * R552;
R854 = R854 + R858 + R859 + R860 + R826;
R858 = A[28] * R854;
R854 = R377 * R793 * R552;
R859 = R163 * R798 * R552;
R860 = R72 * R807 * R552;
R861 = R38 * R816 * R552;
R854 = R854 + R859 + R860 + R861 + R826;
R859 = A[29] * R854;
R854 = R378 * R793 * R552;
R860 = R164 * R798 * R552;
R861 = R73 * R807 * R552;
R862 = R39 * R816 * R552;
R854 = R854 + R860 + R861 + R862 + R826;
R860 = A[30] * R854;
R854 = R379 * R793 * R552;
R861 = R165 * R798 * R552;
R862 = R74 * R807 * R552;
R863 = R40 * R816 * R552;
R854 = R854 + R861 + R862 + R863 + R826;
R861 = A[31] * R854;
R854 = R380 * R793 * R552;
R862 = R166 * R798 * R552;
R863 = R75 * R807 * R552;
R864 = R41 * R816 * R552;
R854 = R854 + R862 + R863 + R864 + R826;
R862 = A[32] * R854;
R854 = R381 * R793 * R552;
R863 = R167 * R798 * R552;
R864 = R76 * R807 * R552;
R865 = R42 * R816 * R552;
R854 = R854 + R863 + R864 + R865 + R826;
R863 = A[33] * R854;
R854 = R382 * R793 * R552;
R864 = R168 * R798 * R552;
R865 = R77 * R807 * R552;
R848 = R43 * R816 * R552;
R854 = R854 + R864 + R865 + R848 + R826;
R864 = A[18] * R854;
R854 = R383 * R793 * R552;
R865 = R169 * R798 * R552;
R848 = R78 * R807 * R552;
R866 = R44 * R816 * R552;
R854 = R854 + R865 + R848 + R866 + R826;
R865 = A[19] * R854;
R854 = R384 * R793 * R552;
R848 = R170 * R798 * R552;
R866 = R79 * R807 * R552;
R850 = R45 * R816 * R552;
R854 = R854 + R848 + R866 + R850 + R826;
R848 = A[20] * R854;
R854 = R385 * R793 * R552;
R866 = R171 * R798 * R552;
R850 = R80 * R807 * R552;
R867 = R46 * R816 * R552;
R854 = R854 + R866 + R850 + R867 + R826;
R866 = A[21] * R854;
R854 = R386 * R793 * R552;
R850 = R172 * R798 * R552;
R867 = R81 * R807 * R552;
R852 = R47 * R816 * R552;
R854 = R854 + R850 + R867 + R852 + R826;
R850 = A[22] * R854;
R854 = R387 * R793 * R552;
R867 = R173 * R798 * R552;
R852 = R82 * R807 * R552;
R868 = R48 * R816 * R552;
R854 = R854 + R867 + R852 + R868 + R826;
R867 = A[23] * R854;
R854 = R388 * R793 * R552;
R852 = R174 * R798 * R552;
R868 = R83 * R807 * R552;
R869 = R49 * R816 * R552;
R854 = R854 + R852 + R868 + R869 + R826;
R852 = A[24] * R854;
R854 = R389 * R793 * R552;
R868 = R175 * R798 * R552;
R869 = R84 * R807 * R552;
double R870 = R50 * R816 * R552;
R854 = R854 + R868 + R869 + R870 + R826;
R868 = A[25] * R854;
R855 = R855 + R856 + R857 + R858 + R859 + R860 + R861 + R862 + R863 + R864 + R865 + R848 + R866 + R850 + R867 + R852 + R868;
R856 = R303 * R529 * R552;
R857 = R159 * R831 * R552;
R858 = R68 * R836 * R552;
R859 = R34 * R841 * R552;
R856 = R856 + R857 + R858 + R859 + R847;
R857 = A[17] * R856;
R856 = R374 * R529 * R552;
R858 = R160 * R831 * R552;
R859 = R69 * R836 * R552;
R860 = R35 * R841 * R552;
R856 = R856 + R858 + R859 + R860 + R847;
R858 = A[26] * R856;
R856 = R375 * R529 * R552;
R859 = R161 * R831 * R552;
R860 = R70 * R836 * R552;
R861 = R36 * R841 * R552;
R856 = R856 + R859 + R860 + R861 + R847;
R859 = A[27] * R856;
R856 = R376 * R529 * R552;
R860 = R162 * R831 * R552;
R861 = R71 * R836 * R552;
R862 = R37 * R841 * R552;
R856 = R856 + R860 + R861 + R862 + R847;
R860 = A[28] * R856;
R856 = R377 * R529 * R552;
R861 = R163 * R831 * R552;
R862 = R72 * R836 * R552;
R863 = R38 * R841 * R552;
R856 = R856 + R861 + R862 + R863 + R847;
R861 = A[29] * R856;
R856 = R378 * R529 * R552;
R862 = R164 * R831 * R552;
R863 = R73 * R836 * R552;
R864 = R39 * R841 * R552;
R856 = R856 + R862 + R863 + R864 + R847;
R862 = A[30] * R856;
R856 = R379 * R529 * R552;
R863 = R165 * R831 * R552;
R864 = R74 * R836 * R552;
R865 = R40 * R841 * R552;
R856 = R856 + R863 + R864 + R865 + R847;
R863 = A[31] * R856;
R856 = R380 * R529 * R552;
R864 = R166 * R831 * R552;
R865 = R75 * R836 * R552;
R848 = R41 * R841 * R552;
R856 = R856 + R864 + R865 + R848 + R847;
R864 = A[32] * R856;
R856 = R381 * R529 * R552;
R865 = R167 * R831 * R552;
R848 = R76 * R836 * R552;
R866 = R42 * R841 * R552;
R856 = R856 + R865 + R848 + R866 + R847;
R865 = A[33] * R856;
R856 = R382 * R529 * R552;
R848 = R168 * R831 * R552;
R866 = R77 * R836 * R552;
R850 = R43 * R841 * R552;
R856 = R856 + R848 + R866 + R850 + R847;
R848 = A[18] * R856;
R856 = R383 * R529 * R552;
R866 = R169 * R831 * R552;
R850 = R78 * R836 * R552;
R867 = R44 * R841 * R552;
R856 = R856 + R866 + R850 + R867 + R847;
R866 = A[19] * R856;
R856 = R384 * R529 * R552;
R850 = R170 * R831 * R552;
R867 = R79 * R836 * R552;
R852 = R45 * R841 * R552;
R856 = R856 + R850 + R867 + R852 + R847;
R850 = A[20] * R856;
R856 = R385 * R529 * R552;
R867 = R171 * R831 * R552;
R852 = R80 * R836 * R552;
R868 = R46 * R841 * R552;
R856 = R856 + R867 + R852 + R868 + R847;
R867 = A[21] * R856;
R856 = R386 * R529 * R552;
R852 = R172 * R831 * R552;
R868 = R81 * R836 * R552;
R854 = R47 * R841 * R552;
R856 = R856 + R852 + R868 + R854 + R847;
R852 = A[22] * R856;
R856 = R387 * R529 * R552;
R868 = R173 * R831 * R552;
R854 = R82 * R836 * R552;
R869 = R48 * R841 * R552;
R856 = R856 + R868 + R854 + R869 + R847;
R868 = A[23] * R856;
R856 = R388 * R529 * R552;
R854 = R174 * R831 * R552;
R869 = R83 * R836 * R552;
R870 = R49 * R841 * R552;
R856 = R856 + R854 + R869 + R870 + R847;
R854 = A[24] * R856;
R856 = R389 * R529 * R552;
R869 = R175 * R831 * R552;
R870 = R84 * R836 * R552;
double R871 = R50 * R841 * R552;
R856 = R856 + R869 + R870 + R871 + R847;
R869 = A[25] * R856;
R857 = R857 + R858 + R859 + R860 + R861 + R862 + R863 + R864 + R865 + R848 + R866 + R850 + R867 + R852 + R868 + R854 + R869;
output[0]=R849;
output[1]=R851;
output[2]=R853;
output[3]=R855;
output[4]=R857;
//T(R1)0 ={ R849, R851, R853, R855, R857 };
//Return;
}

__device__ __host__ inline double Square(double &x)
{
    return x*x;
}
__device__ __host__ inline double Reciprocal( double &x)
{
    return 1./x;
}
__device__ __host__ inline double Power3(double &x)
{
    return x*x*x;
}

__device__ __host__ inline double Power4(double &x)
{
    return x*x*x*x;
}
