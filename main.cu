```cuda
    #include <cuda.h>
    #include <cstdlib>
    #include <cstdio>
    #include <omp.h>
    #include <cuda_runtime.h>

    using namespace std;

    #include "utils.cu"
    #include "kernelACGPU.cu"

    void cpu_sim(int n, int pasos, bool *board, bool *temp, int nt){
        int i,j,neighbour_live_cell;
        double t;
        omp_set_num_threads(nt);
        for(int p=0; p<pasos; ++p){
                    printf("[AC][CPU][%i]\n", p);
                    t = omp_get_wtime();

                    #pragma omp parallel for
                    for(i=0; i<n; i++){
                        for(j=0;j<n;j++){
                            neighbour_live_cell=count_live_neighbour_cell(board,n,i,j);
                            if(board[i*n+j] && (neighbour_live_cell==2 || neighbour_live_cell==3)){
                                temp[i*n+j]=1;
                            }
                            else if((!board[i*n+j]) && neighbour_live_cell==3){
                                temp[i*n+j]=1;
                            }
                            else{
                            temp[i*n+j]=0;
                            }
                        }
                    }
                
                    
                    std::swap(board,temp);
                    if(n<=128) printAC(n,board);
                    printf("done in %f[s]\n", omp_get_wtime() - t);
                    printf("Press enter to continue\n");
                    fflush(stdout);
                    getchar();
        }
    }

    void gpu_sim(int n,int pasos,bool *board,bool *temp, int nb, int GPUID){
        bool *board_d, *temp_d;
        float msecs;
        cudaEvent_t start, stop;
        cudaEventCreate(&start); cudaEventCreate(&stop);
        cudaSetDevice(GPUID);
        cudaMalloc(&board_d,sizeof(bool)*n*n);
        cudaMalloc(&temp_d,sizeof(bool)*n*n);
        cudaMemcpy(board_d,board,sizeof(bool)*n*n, cudaMemcpyHostToDevice);
        dim3 block(nb,nb);
        dim3 grid((n+block.x-1)/block.x, (n+block.y-1)/block.y);
        for(int p=0; p<pasos; p++){
            printf("[AC][GPU][%i]\n", p);
            cudaEventRecord(start);
            GoLKernel<<<grid, block>>>(board_d, temp_d, n);
            cudaDeviceSynchronize();    cudaEventRecord(stop);  cudaEventSynchronize(stop);
            cudaEventElapsedTime(&msecs,start,stop);
            cudaMemcpy(board_d,temp_d,sizeof(bool)*n*n,cudaMemcpyDeviceToDevice);
            cudaMemcpy(board,board_d,sizeof(bool)*n*n,cudaMemcpyDeviceToHost);
            if(n<=128) printAC(n,board);
             printf("done in %f[s]\n", msecs/1000.0f);
            printf("Press enter to continue\n");
            fflush(stdout);
            getchar();
        }

    }

    int main(int argc, char **argv){
        if(argc != 8){
            fprintf(stderr, "Error. Debe ejecutarse como ./prog <gpu-id> <n> <seed> <pasos> <nt> <nb> <cpu-o-gpu>\n\n");
            exit(EXIT_FAILURE);
        }
        long GPUID       = atoi(argv[1]);
        int n      = atoi(argv[2]);
        int seed     = atoi(argv[3]);
        int pasos    = atoi(argv[4]);
        int nt   = atoi(argv[5]);
        int nb  = atoi(argv[6]);
        int CPUoGPU = atoi(argv[7]);

        srand(seed);
        
        bool *board = (bool*)malloc(n*n*sizeof(bool));
        bool *temp = (bool*)malloc(n*n*sizeof(bool));
        for(int i=0; i<n; i++){
            for(int j=0; j<n; j++)
                board[i*n+j] = rand()%2;
        }
        printf("[AC][ORIGINAL]\n");
        printAC(n,board);
        printf("Press enter to continue\n");
        fflush(stdout);
        getchar();
        if(CPUoGPU){
            //modo GPU
            gpu_sim(n,pasos,board,temp,nb,GPUID);
        }
        else{
            //modo CPU
        cpu_sim(n,pasos,board,temp, nt);
        }
    }

```