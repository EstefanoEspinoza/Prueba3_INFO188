#pragma once
__global__ void kernel_CA2D(int n, int *in, int *out){
    int tx = blockIdx.x*blockDim.x + threadIdx.x;
    int ty = blockIdx.y*blockDim.y + threadIdx.y;
    if(tx >= n || ty >= n){
        return;
    }

    int yup = ty == 0? n-1 : ty-1;
    int ydown = (ty+1)%n;
    int xleft = tx == 0? n-1 : tx-1;
    int xright = (tx+1)%n;
    
    int neighborhood = in[(yup)*n + (xleft)] + in[(yup)*n + tx] + in[(yup)*n + xright] + 
                       in[ty*n + xleft]      +        0         + in[ty*n + xright] +
                       in[(ydown)*n + (xleft)] + in[(ydown)*n + tx] + in[(ydown)*n + xright];

    int cell = in[ty*n + tx];
    out[ty*n + tx] = 0;
    if(cell && (neighborhood == 2 || neighborhood == 3)){
        out[ty*n + tx] = 1;
    }
    if(!cell && (neighborhood == 3)){
        out[ty*n + tx] = 1;
    }
}