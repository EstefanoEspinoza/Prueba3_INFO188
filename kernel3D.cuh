#pragma once
__global__ void kernel_CA3D(int n, int *in, int *out){
    int tx = blockIdx.x*blockDim.x + threadIdx.x;
    int ty = blockIdx.y*blockDim.y + threadIdx.y;
    int tz = blockIdx.z*blockDim.z + threadIdx.z;
   
    if(tx >= n || ty >= n || tz >= n){
        return;
    }

    int yup = ty == 0? n-1 : ty-1;
    int ydown = (ty+1)%n;
    int xleft = tx == 0? n-1 : tx-1;
    int xright = (tx+1)%n;
    
    int back = tz == 0? n-1 : tz-1;
    int front = (tz+1)%n;

    // trasformaciones
    int fr = front*n*n;
    int bk = back*n*n;
    int tzn = tz*n*n;

    // capa trasera
    int neighborhoodBack =  in[bk + (yup)*n + (xleft)]   + in[bk + (yup)*n + tx]   + in[bk + (yup)*n + xright] + 
                            in[bk + ty*n + xleft]        +   in[bk + ty*n + tx]    + in[bk + ty*n + xright] +
                            in[bk + (ydown)*n + (xleft)] + in[bk + (ydown)*n + tx] + in[bk + (ydown)*n + xright];
    // capa intermedia
    int neighborhoodMiddle= in[tzn + (yup)*n + (xleft)]   + in[tzn + (yup)*n + tx]    + in[tzn +(yup)*n + xright] + 
                            in[tzn + ty*n + xleft]        +        0                     + in[tzn + ty*n + xright] +
                            in[tzn + (ydown)*n + (xleft)] + in[tzn + (ydown)*n + tx]  + in[tzn + (ydown)*n + xright];
    // capa frontal
    int neighborhoodFront = in[fr + (yup)*n + (xleft)]   + in[fr + (yup)*n + tx]   + in[fr + (yup)*n + xright] + 
                            in[fr + ty*n + xleft]        +   in[fr + ty*n + tx]    + in[fr + ty*n + xright] +
                            in[fr + (ydown)*n + (xleft)] + in[fr + (ydown)*n + tx] + in[fr + (ydown)*n + xright];
   
    int neighborhood = neighborhoodMiddle + neighborhoodBack + neighborhoodFront;
    int cell = in[tzn + ty*n + tx];
    out[tzn + ty*n + tx] = 0;

    if(cell && (neighborhood == 2 || neighborhood == 3)){out[tzn + ty*n + tx] = 1;}

    if(!cell && (neighborhood == 3)){out[tzn + ty*n + tx] = 1;}
}