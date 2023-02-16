#include <iomanip>
#include <iostream>
#include <chrono>

#include "output.hpp"

using namespace std::chrono;

using std::cout;
using std::endl;
using std::fixed;
using std::flush;
using std::setprecision;
using std::setw;

void printMatrix(int rows, int cols, double ** matrix) {
    for(int row = 0; row < rows; row++) {
        for(int col = 0; col < cols; col++) {
            cout << fixed << setw(12) << setprecision(2) << matrix[row][col] << flush;
        }

        cout << endl << flush;
    }

    cout << endl << flush;
}

void printStatistics(int threads, long runtime_seq, long runtime_par) {
    double acceleration = 1.0 * runtime_seq / runtime_par;
    double efficiency = acceleration / threads;

    cout << "Runtime sequential: " << runtime_seq / 1000000.0 << " seconds" << endl << flush;
    cout << "Runtime parallel  : " << runtime_par / 1000000.0 << " seconds" << endl << flush;
    cout << "Acceleration      : " << acceleration << endl << flush;
    cout << "Efficiency        : " << efficiency << endl << flush;
}

// Ne pas toucher les fonctions ci-dessous !
int tostr(int nbr, char **str) {
    int len = snprintf(NULL, 0, "%d", nbr);

    *str = (char *)malloc(len + 1);
    return snprintf(*str, len + 1, "%d", nbr);
}

void saveResults(int threads, int rows, int cols, int iterations, double **matrix, long runtime_seq, long runtime_par) {
    FILE *file;

    char 
        *sNbThreads = NULL, 
        *sProblem = NULL, 
        *sCols = NULL, 
        *sIterations = NULL;

    int 
        lenNbThreads = tostr(threads, &sNbThreads),
        lenRows = tostr(rows, &sProblem),
        lenCols = tostr(cols, &sCols),
        lenIterations = tostr(iterations, &sIterations);

    char *fileName = (char *) malloc(lenNbThreads + lenRows + lenCols + lenIterations + 1);
    sprintf(fileName, "matrix_%s%s%s%s.txt", sNbThreads, sProblem, sCols, sIterations);

    printf("Saving results to %s\n", fileName);

    free(sNbThreads);
    free(sProblem);
    free(sCols);
    free(sIterations);

    file = fopen(fileName, "w+");
    
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            fprintf(file, "%.2f ", matrix[i][j]);
        }

        fprintf(file, "\n");
    }

    fprintf(file, "\n");

    double acceleration = 1.0 * runtime_seq / runtime_par;
    double efficiency = acceleration / threads;
    
    fprintf(file, "%.6f\n", runtime_seq / 1000000.0);
    fprintf(file, "%.6f\n", runtime_par / 1000000.0);
    fprintf(file, "%.2f\n", acceleration);
    fprintf(file, "%.2f\n", efficiency);

    fclose(file);
    free(fileName);
}