#include "matrix.hpp"

double ** allocateMatrix(int rows, int cols) {
    double ** matrix = new double*[rows];

    for(int i = 0; i < rows; i++) {
        matrix[i] = new double[cols];
    }

    return matrix;
}

double ** transposeMatrix(double **matrix, int rows, int cols) {
    // Allocate memory for the transposed matrix
    double **transposed = allocateMatrix(cols, rows);

    // Transpose the matrix
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            transposed[j][i] = matrix[i][j];
        }
    }

    return transposed;
}

void deallocateMatrix(int rows, double ** matrix) {
    for(int i = 0; i < rows; i++) {
        delete(matrix[i]);
        matrix[i] = nullptr;
    }

    delete(matrix);
    matrix = nullptr;
}

void fillMatrix(int rows, int cols, double ** matrix) {
     for(int row = 0; row < rows; row++) {
        for(int col = 0; col < cols; col++) {
            matrix[row][col] = row * (rows - row - 1) * col * (cols - col - 1);
        }
    }
}
