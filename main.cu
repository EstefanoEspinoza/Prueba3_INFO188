    #pragma once
    
    #include <cuda.h>
    #include <cstdlib>
    #include <cstdio>
    #include <omp.h>
    #include <cuda_runtime.h>

    using namespace std;

    //#include "utils.cu"
    //#include "kernelGPU_AC.cu"

    // -----------------------------------------------------------------------------------------------

    #pragma once
__global__ void GoLKernel(bool *tablero, bool *temp, int n) {
    int fila = blockIdx.y * blockDim.y + threadIdx.y;
    int columna = blockIdx.x * blockDim.x + threadIdx.x;

    if (fila < 0 || fila >= n || columna < 0 || columna >= n) {
        return;
    }

    int cont = 0;
    for (int i = -1; i <= 1; i++) {
        for (int j = -1; j <= 1; j++) {
            if (i == 0 && j == 0) {
                continue;
            }

            int vecindad_eje_X = fila + i;
            int vecindad_eje_y = columna + j;
            if (vecindad_eje_X < 0 || vecindad_eje_X >= n || vecindad_eje_y < 0 || vecindad_eje_y >= n) {
                continue;
            }

            if (tablero[(vecindad_eje_X * n) + vecindad_eje_y])
                cont++;
        }
    }

    if (tablero[(fila * n) + columna]) {
        if (cont == 2 || cont == 3) {
            temp[(fila * n) + columna] = 1;
        } else {
            temp[(fila * n) + columna] = 0;
        }
    } else {
        if (cont == 3) {
            temp[(fila * n) + columna] = 1;
        } else {
            temp[(fila * n) + columna] = 0;
        }
    }
}

    // -----------------------------------------------------------------------------------------------

    void printAutomataCelular(int n,bool *board){
     for(int i = 0; i < n; i++){
        for(int j = 0; j < n; j++) {
            printf(board[i*n+j] ? "*" : "/");
        }
        printf("\n");
    }
}


int count_live_neighbour_cell(bool *board,int n, int r, int c){
    int i, j, count=0;
    for(i = r - 1; i < = r + 1; i++){
        for(j = c - 1; j < = c + 1; j++){
            if((i==r && j==c) || (i<0 || j<0) || (i>=n || j>=n)){
                continue;
            }
            if(board[i*n+j]){
                count++;
            }
        }
    }
    return count;
}

    // -----------------------------------------------------------------------------------------------

    void cpu_sim(int n, int pasos, bool *board, bool *temp, int nt){
        int neighbour_live_cell;
        double t,f;
        omp_set_num_threads(nt);
        for(int p = 0; p < pasos; p++){
                    printf("[AC][CPU] paso %i, n %i, nt %i\n", p,n,nt);
                    t = omp_get_wtime();

                    #pragma omp parallel for
                    for(int i = 0 ; i < n; i++){
                        for(int j = 0; j < n; j++){
                            neighbour_live_cell=count_live_neighbour_cell(board,n,i,j);
                            if(board[(i * n) + j] && (neighbour_live_cell == 2 || neighbour_live_cell == 3)){
                                temp[(i * n) + j] = 1;
                            }
                            else if((!board[(i * n) + j]) && neighbour_live_cell == 3){
                                temp[(i * n) + j] = 1;
                            }
                            else{
                            temp[(i * n) + j] = 0;
                            }
                        }
                    }
                
                    f = omp_get_wtime() - t;
                    std::swap(board,temp);
                    if(n<=128){
                        printAutomataCelular(n,board);
                    }
                    printf("terminado en %f[s]\n", f);
                    printf("Presiona enter para continuar\n");
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
        dim3 grid((n + block.x - 1)/block.x, (n + block.y - 1)/block.y);
        for(int p = 0; p < pasos; p++){
            printf("[AC][GPU] paso %i, n %i, nb %i\n", p,n,nb);
            cudaEventRecord(start);
            GoLKernel<<<grid, block>>>(board_d, temp_d, n);
            cudaDeviceSynchronize();    cudaEventRecord(stop);  cudaEventSynchronize(stop);
            cudaEventElapsedTime(&msecs,start,stop);
            cudaMemcpy(board_d,temp_d,sizeof(bool)*n*n,cudaMemcpyDeviceToDevice);
            if(n<=128){
                cudaMemcpy(board,board_d,sizeof(bool)*n*n,cudaMemcpyDeviceToHost);
                printAutomataCelular(n,board);
            }
            printf("terminado en %f[s]\n", msecs/1000.0f);
            printf("Presione enter para continuar\n");
            fflush(stdout);
            getchar();
        }
        if(n>128){
            cudaMemcpy(board,board_d,sizeof(bool)*n*n,cudaMemcpyDeviceToHost);
            }
    }

    int main(int argc, char **argv){
        if(argc != 8){
            fprintf(stderr, "Error. Debe ejecutarse como ./prog Id_GPU(0,1,2) n semilla pasos Numero_Threads Numero_Bloques cpu = 0 o gpu = 1 \n\n");
            exit(EXIT_FAILURE);
        }
        long GPUID       = atoi(argv[1]);
        int n      = atoi(argv[2]);
        int seed     = atoi(argv[3]);
        int pasos    = atoi(argv[4]);
        int nt   = atoi(argv[5]);
        int nb  = atoi(argv[6]); // si el valor ingresado en nb es 0 lanza core dumped xd
        int CPU_o_GPU = atoi(argv[7]);

        srand(seed);
        
        bool *board = (bool*)malloc(n*n*sizeof(bool));
        bool *temp = (bool*)malloc(n*n*sizeof(bool));
        for(int i = 0; i < n; i++){
            for(int j = 0; j < n; j++)
                board[(i * n) + j] = rand()%2;
        }
        
        printf("[AC][ORIGINAL] created\n");
        if(n<=128){
            printAutomataCelular(n,board);  //Imprime la primera iteración de la ejecución.
        } 
        printf("Presiona enter para continuar\n");
        fflush(stdout);
        getchar();
        if(CPU_o_GPU){
            //entra al modo GPU para esta ejecución
            gpu_sim(n,pasos,board,temp,nb,GPUID);
        }
        else{
            //entra al modo CPU para esta ejecución
        cpu_sim(n,pasos,board,temp, nt);
        }
    }
