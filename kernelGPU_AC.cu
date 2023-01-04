__global__ void GoLKernel(bool *board, bool *temp, int n) {
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

            int vecino_fila = fila + i;
            int vecino_columna = columna + j;
            if (vecino_fila < 0 || vecino_fila >= n || vecino_columna < 0 || vecino_columna >= n) {
                continue;
            }
            if (board[vecino_fila*n+vecino_columna])
                cont++;
        }
    }
    if (board[fila*n+col]) {
        if (cont == 2 || cont == 3) {
            temp[fila*n+columna] = 1;
        } else {
            temp[fila*n+columna] = 0;
        }
    } else {
        if (cont == 3) {
            temp[fila*n+columna] = 1;
        } else {
            temp[fila*n+columna] = 0;
        }
    }
}#pragma once
__global__ void GoLKernel(bool *board, bool *temp, int n) {
    int fila = blockIdx.y * blockDim.y + threadIdx.y;
    int columna = blockIdx.x * blockDim.x + threadIdx.x;

    if (fila < 0 || fila >= n || col < 0 || col >= n) {
        return;
    }
    int cont = 0;
    for (int i = -1; i <= 1; i++) {
        for (int j = -1; j <= 1; j++) {
            if (i == 0 && j == 0) {
                continue;
            }

            int vecino_fila = fila + i;
            int vecino_columna = columna + j;
            if (vecino_fila < 0 || vecino_fila >= n || vecino_columna < 0 || vecino_columna >= n) {
                continue;
            }

            if (board[vecino_fila*n+vecino_columna])
                cont++;
        }
    }

    if (board[fila*n+columna]) {
        if (cont == 2 || cont == 3) {
            temp[fila*n+columna] = 1;
        } else {
            temp[fila*n+columna] = 0;
        }
    } else {
        if (cont == 3) {
            temp[fila*n+columna] = 1;
        } else {
            temp[fila*n+columna] = 0;
        }
    }
}
