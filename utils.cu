
#pragma once
#include <cuda.h>
#include <cstdlib>
#include <cstdio>
#include <omp.h>

void printAutomataCelular(int n,bool *board){
     for(int i=0; i<n; i++){
        for(int j=0; j<n; j++) printf(board[i*n+j] ? "*" : "/");
        printf("\n");
    }
}


int count_live_neighbour_cell(bool *board,int n, int r, int c){
    int i, j, count=0;
    for(i=r-1; i<=r+1; i++){
        for(j=c-1;j<=c+1;j++){
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

//Función sacada directamente desde el enlace que dejó en el .txt de la prueba 3, puesta por recomendación de un compañero.