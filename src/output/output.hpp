#ifndef OUTPUT_HPP
#define OUTPUT_HPP

void printMatrix(int rows, int cols, double ** matrix);
void printStatistics(int threads, long runtime_seq, long runtime_par);

int tostr(int nbr, char **str);
void saveResults(int threads, int rows, int cols, int iterations, double **matrix, long runtime_seq, long runtime_par);

#endif
