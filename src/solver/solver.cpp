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

    double * linePrevBuffer = new double[cols];
    double * lineCurrBuffer = new double[cols];

    double * previousBufferReceive = new double[cols];
    double * nextBufferReceive = new double[cols];

    double * firstLine = new double[cols];
    double * secondLine = new double[cols];
    double * beforeLastLine = new double[cols];
    double * lastLine = new double[cols];


    for(int k = 0; k < iterations; k++) {
        memcpy(firstLine, matrix[0], cols * sizeof(double));
        memcpy(secondLine, matrix[1], cols * sizeof(double));
        memcpy(beforeLastLine, matrix[rows-2], cols * sizeof(double));
        memcpy(lastLine, matrix[rows-1], cols * sizeof(double));

        // Send lines to other threads
        if (rank == 0)
        {
            MPI_Request request;
            MPI_Isend(lastLine, cols, MPI_DOUBLE, rank + 1, 0, MPI_COMM_WORLD, &request);
        }
        else if (rank == lastRank)
        {
            MPI_Request request;
            MPI_Isend(firstLine, cols, MPI_DOUBLE, rank - 1, 0, MPI_COMM_WORLD, &request);

        }
        else
        {
            // Send the first line of the previous iteration to rank - 1 
            // Send the last line of the previous iteration to rank + 1
            MPI_Request request1, request2;
            MPI_Isend(firstLine, cols, MPI_DOUBLE, rank - 1, 0, MPI_COMM_WORLD, &request1);
            MPI_Isend(lastLine, cols, MPI_DOUBLE, rank + 1, 0, MPI_COMM_WORLD, &request2);

        }

        // ComputeMiddle in each case, independently
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

        // Edge cases
        // Solve last line with received buffer
        if (rank == 0)
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
       
        else if (rank == lastRank)
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
        // Solve first AND last line with received buffer 
        else
        {
            // Receive line from rank - 1 to compute first line
            MPI_Recv(previousBufferReceive, cols , MPI_DOUBLE, rank - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            for(int j = 1; j < cols - 1; j++) {
                c = firstLine[j];
                t = previousBufferReceive[j];
                b = secondLine[j];
                l = firstLine[j - 1];
                r = firstLine[j + 1];

                sleep_for(microseconds(sleep));
                matrix[0][j] = c * (1.0 - 4.0 * td / h_square) + (t + b + l + r) * (td / h_square);
            }

            // Receive line from rank + 1 to compute last line
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





