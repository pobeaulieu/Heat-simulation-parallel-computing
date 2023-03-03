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
    double * lineNextOutBuffer = new double[cols];

    for(int k = 0; k < iterations; k++) {

        if (rank == 0){
            // Send the last line of the previous iteration to rank + 1
            MPI_Request request;
            MPI_Isend(matrix[rows-1], cols, MPI_DOUBLE, rank + 1, 0, MPI_COMM_WORLD, &request);
            //MPI_Send(matrix[rows-1], cols, MPI_DOUBLE, rank + 1, 0, MPI_COMM_WORLD);
           
            // Fill previous buffer
            memcpy(linePrevBuffer, matrix[0], cols * sizeof(double));

            //Compute middle
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
            
            //Receive line from rank + 1 to compute last line
            MPI_Recv(lineNextOutBuffer, cols, MPI_DOUBLE, rank + 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            for(int j = 1; j < cols - 1; j++) {
                    c = lineCurrBuffer[j];
                    t = linePrevBuffer[j];
                    b = lineNextOutBuffer[j];
                    l = lineCurrBuffer[j - 1];
                    r = lineCurrBuffer[j + 1];

                    sleep_for(microseconds(sleep));
                    matrix[rows-1][j] = c * (1.0 - 4.0 * td / h_square) + (t + b + l + r) * (td / h_square);
            }
        }
        else if (rank == lastRank){
            // Send the first line of the previous iteration to rank - 1
            MPI_Request request;
            MPI_Isend(matrix[0], cols, MPI_DOUBLE, rank - 1, 0, MPI_COMM_WORLD, &request);
            //MPI_Send(matrix[0], cols, MPI_DOUBLE, rank - 1, 0, MPI_COMM_WORLD);


            // Fill previous buffer - Receive line from rank - 1
            MPI_Recv(linePrevBuffer, cols, MPI_DOUBLE, rank - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            // Compute using the received line
            for(int i = 0; i < rows - 1; i++) {

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
        else{
            // Send the first line of the previous iteration to rank - 1 
            // Send the last line of the previous iteration to rank + 1
            MPI_Request request1, request2;
            MPI_Isend(matrix[0], cols, MPI_DOUBLE, rank - 1, 0, MPI_COMM_WORLD, &request1);
            MPI_Isend(matrix[rows-1], cols, MPI_DOUBLE, rank + 1, 0, MPI_COMM_WORLD, &request2);
            //MPI_Send(matrix[0], cols, MPI_DOUBLE, rank - 1, 0, MPI_COMM_WORLD);
            //MPI_Send(matrix[rows-1], cols, MPI_DOUBLE, rank + 1, 0, MPI_COMM_WORLD);
        
            //Receive line from rank - 1 to compute first line
            MPI_Recv(linePrevBuffer, cols , MPI_DOUBLE, rank - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            
            for(int i = 0; i < rows - 1; i++) {

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

            //Receive line from rank + 1 to compute last line
            MPI_Recv(lineNextOutBuffer, cols, MPI_DOUBLE, rank + 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            for(int j = 1; j < cols - 1; j++) {
                    c = matrix[rows-1][j];
                    t = linePrevBuffer[j];
                    b = lineNextOutBuffer[j];
                    l = matrix[rows-1][j - 1];
                    r = matrix[rows-1][j + 1];


                    sleep_for(microseconds(sleep));
                    matrix[rows-1][j] = c * (1.0 - 4.0 * td / h_square) + (t + b + l + r) * (td / h_square);
            }
        }
    }

}

void printBuffer(int cols, double * buffer) {
        for(int col = 0; col < cols; col++) {
            cout << fixed << buffer[col] << " " << flush;
        }

        cout << endl << flush;
}





