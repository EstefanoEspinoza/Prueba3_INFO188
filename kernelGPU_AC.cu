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