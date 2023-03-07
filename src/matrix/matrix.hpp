#ifndef MATRIX_HPP
#define MATRIX_HPP

double ** allocateMatrix(int rows, int cols);
void deallocateMatrix(int rows, double ** matrix);
double ** transposeMatrix(double **matrix, int rows, int cols); 
void fillMatrix(int rows, int cols, double ** matrix);

#endif
