#pragma once
__global__ void GoLKernel(bool *board, bool *temp, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < 0 || row >= n || col < 0 || col >= n) {
        return;
    }

    int cont = 0;
    for (int i = -1; i <= 1; i++) {
        for (int j = -1; j <= 1; j++) {
            if (i == 0 && j == 0) {
                continue;
            }

            int vecino_x = row + i;
            int vecino_y = col + j;
            if (vecino_x < 0 || vecino_x >= n || vecino_y < 0 || vecino_y >= n) {
                continue;
            }

            if (board[vecino_x*n+vecino_y])
                cont++;
        }
    }

    if (board[row*n+col]) {
        if (cont == 2 || cont == 3) {
            temp[row*n+col] = 1;
        } else {
            temp[row*n+col] = 0;
        }
    } else {
        if (cont == 3) {
            temp[row*n+col] = 1;
        } else {
            temp[row*n+col] = 0;
        }
    }
}