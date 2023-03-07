#include <chrono>
#include <cstring>
#include <thread>

#include <mpi.h>

#include "solver.hpp"
#include "../matrix/matrix.hpp"
#include "../output/output.hpp"

#include <iostream>

using std::memcpy;

using std::this_thread::sleep_for;
using std::chrono::microseconds;
using std::cout;
using std::endl;
using std::fixed;
using std::flush;

void printBuffer(int cols, double * buffer);

void solveSeq(int rows, int cols, int iterations, double td, double h, int sleep, double ** matrix)
{
    double c, l, r, t, b;
    
    double h_square = h * h;

    double * linePrevBuffer = new double[cols];
    double * lineCurrBuffer = new double[cols];

    for(int k = 0; k < iterations; k++) {

        memcpy(linePrevBuffer, matrix[0], cols * sizeof(double));
        for(int i = 1; i < rows - 1; i++) {

            memcpy(lineCurrBuffer, matrix[i], cols * sizeof(double));
            for(int j = 1; j < cols - 1; j++) {
                c = lineCurrBuffer[j];
                t = linePrevBuffer[j];
                b = matrix[i + 1][j];
                l = lineCurrBuffer[j - 1];
                r = lineCurrBuffer[j + 1];


                sleep_for(microseconds(sleep));
                matrix[i][j] = c * (1.0 - 4.0 * td / h_square) + (t + b + l + r) * (td / h_square);
            }

            memcpy(linePrevBuffer, lineCurrBuffer, cols * sizeof(double));
        }
    }
}

void solvePar(int rows, int cols, int iterations, double td, double h, int sleep, double ** matrix, int rank, int lastRank) 
{
    double c, l, r, t, b;
    double h_square = h * h;

    // Communication buffers
    double * previousBufferReceive = new double[cols];
    double * nextBufferReceive = new double[cols];

    // Calculation buffers
    double * linePrevBuffer = new double[cols];
    double * lineCurrBuffer = new double[cols];

    // Previous iteration buffers. Useful to compute middle first
    double * firstLine = new double[cols];
    double * secondLine = new double[cols];
    double * beforeLastLine = new double[cols];
    double * lastLine = new double[cols];

    for(int k = 0; k < iterations; k++) {
        // Copy borders to compute middle first
        memcpy(firstLine, matrix[0], cols * sizeof(double));
        memcpy(secondLine, matrix[1], cols * sizeof(double));
        memcpy(beforeLastLine, matrix[rows-2], cols * sizeof(double));
        memcpy(lastLine, matrix[rows-1], cols * sizeof(double));

        // Send buffers to other threads as soon as possible
        if (rank != lastRank)
        {
            MPI_Request request;
            MPI_Isend(lastLine, cols, MPI_DOUBLE, rank + 1, 0, MPI_COMM_WORLD, &request);
        }
        if (rank != 0)
        {
            MPI_Request request;
            MPI_Isend(firstLine, cols, MPI_DOUBLE, rank - 1, 0, MPI_COMM_WORLD, &request);
        }
    
        // ComputeMiddle in each case, independently of other threads
        memcpy(linePrevBuffer, firstLine, cols * sizeof(double));
        for(int i = 1; i < rows - 1; i++) {
            memcpy(lineCurrBuffer, matrix[i], cols * sizeof(double));
            for(int j = 1; j < cols - 1; j++) {
                c = lineCurrBuffer[j];
                t = linePrevBuffer[j];
                b = matrix[i + 1][j];
                l = lineCurrBuffer[j - 1];
                r = lineCurrBuffer[j + 1];
                sleep_for(microseconds(sleep));
                matrix[i][j] = c * (1.0 - 4.0 * td / h_square) + (t + b + l + r) * (td / h_square);
            }
            memcpy(linePrevBuffer, lineCurrBuffer, cols * sizeof(double));
        }

        // Compute edge lines
        // Edge case 1 - Last line - Receive the next line if there is a neighbor at rank + 1
        if (rank != lastRank)
        {
            MPI_Recv(nextBufferReceive, cols, MPI_DOUBLE, rank + 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            for(int j = 1; j < cols - 1; j++) {
                c = lastLine[j];
                t = beforeLastLine[j];
                b = nextBufferReceive[j];
                l = lastLine[j - 1];
                r = lastLine[j + 1];
                sleep_for(microseconds(sleep));
                matrix[rows-1][j] = c * (1.0 - 4.0 * td / h_square) + (t + b + l + r) * (td / h_square);
            }
        }
        // Edge case 2 - First line -  Receive the previous line if there is a neighbor at rank - 1
        if (rank != 0)
        {
            MPI_Recv(previousBufferReceive, cols, MPI_DOUBLE, rank - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            for(int j = 1; j < cols - 1; j++) {
                c = firstLine[j];
                t = previousBufferReceive[j];
                b = secondLine[j];
                l = firstLine[j - 1];
                r = firstLine[j + 1];
                sleep_for(microseconds(sleep));
                matrix[0][j] = c * (1.0 - 4.0 * td / h_square) + (t + b + l + r) * (td / h_square);
            }
        }  
    }

    delete[](linePrevBuffer);
    delete[](lineCurrBuffer);
    delete[](previousBufferReceive);
    delete[](firstLine);
    delete[](secondLine);
    delete[](beforeLastLine);
    delete[](lastLine);

}

void printBuffer(int cols, double * buffer) {
        for(int col = 0; col < cols; col++) {
            cout << fixed << buffer[col] << " " << flush;
        }

        cout << endl << flush;
}





