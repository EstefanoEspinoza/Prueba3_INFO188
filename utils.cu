#pragma once
#include <cuda.h>
#include <cstdlib>
#include <cstdio>
#include <omp.h>

void printAC(int n,bool board){
     for(int i=0; i<n; i++){
        for(int j=0; j<n; j++) printf(board[in+j] ? "O" : "░");
        printf("\n");
    }
}

 //funcion auxiliar basada en la disponible en  https://www.geeksforgeeks.org/program-for-conways-game-of-life/
int count_live_neighbour_cell(bool board,int n, int r, int c){
    int i, j, count=0;
    for(i=r-1; i<=r+1; i++){
        for(j=c-1;j<=c+1;j++){
            if((i==r && j==c)  (i<0  j<0)  (i>=n  j>=n)){
                continue;
            }
            if(board[in+j]){
                count++;
            }
        }
    }
    return count;
}