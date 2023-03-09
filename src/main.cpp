#include <iostream>
#include <iomanip>
#include <chrono>
#include <thread>

#include <mpi.h>

#include "matrix/matrix.hpp"
#include "output/output.hpp"
#include "solver/solver.hpp"

using std::cout;
using std::endl;
using std::fixed;
using std::flush;
using std::setprecision;
using std::setw;

void usage();
void command(int argc, char* argv[]);

void initial(int rows, int cols);
long sequential(int rows, int cols, int iters, double td, double h, int sleep);
long parallel(int rows, int cols, int iters, double td, double h, int sleep, int rank, double ** matrix, int procCount);

using namespace std::chrono;

using std::cout;
using std::endl;
using std::flush;
using std::setprecision;
using std::setw;
using std::stod;
using std::stoi;

#include <iostream>
#include <vector>
#include <cmath>


void distributeRows(int procCount, int rows, int cols, int * threadRows, int * displacements);

int main(int argc, char* argv[]) {
    int rows;
    int cols;
    int iters;
    double td;
    double h;

    // MPI variables.
    int mpi_status;
    int rank;
    int procCount;

    // Resolution variables.
    // Sleep will be in microseconds during execution.
    int sleep = 5;

    // Timing variables.
    long runtime_seq = 0;
    long runtime_par = 0;

    if(6 != argc)
    {
        usage();
        return EXIT_FAILURE;
    }

    mpi_status = MPI_Init(&argc, &argv);

    if(MPI_SUCCESS != mpi_status)
    {
        cout << "MPI initialization failure." << endl << flush;
        return EXIT_FAILURE;
    }

    rows = stoi(argv[1], nullptr, 10);
    cols = stoi(argv[2], nullptr, 10);
    iters = stoi(argv[3], nullptr, 10);
    td = stod(argv[4], nullptr);
    h = stod(argv[5], nullptr);

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &procCount);

    if(0 == rank)
    {
        command(argc, argv);
        initial(rows, cols);
        runtime_seq = sequential(rows, cols, iters, td, h, sleep);
    }

    // Ensure that no process will start computing early.
    MPI_Barrier(MPI_COMM_WORLD);

    double ** matrix = allocateMatrix(rows, cols);
    fillMatrix(rows, cols, matrix);

    runtime_par = parallel(rows, cols, iters, td, h, sleep, rank, matrix, procCount);

    if(0 == rank)
    {
        printStatistics(procCount, runtime_seq, runtime_par);
        saveResults(procCount, rows, cols, iters, matrix, runtime_seq, runtime_par);
    }

    deallocateMatrix(rows, matrix);
    mpi_status = MPI_Finalize();

    if(MPI_SUCCESS != mpi_status)
    {
        cout << "Execution finalization terminated in error." << endl << flush;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}

void usage()
{
    cout << "Invalid arguments." << endl << flush;
    cout << "Arguments: m n np td h" << endl << flush;
}

void command(int argc, char* argv[])
{
    cout << "Command:" << flush;

    for(int i = 0; i < argc; i++) {
        cout << " " << argv[i] << flush;
    }

    cout << endl << flush;
}

void initial(int rows, int cols)
{
    double ** matrix = allocateMatrix(rows, cols);
    fillMatrix(rows, cols, matrix);

    cout << "-----  INITIAL   -----" << endl << flush;
    printMatrix(rows, cols, matrix);

    deallocateMatrix(rows, matrix);
}

long sequential(int rows, int cols, int iters, double td, double h, int sleep)
{
    double ** matrix = allocateMatrix(rows, cols);
    fillMatrix(rows, cols, matrix);

    time_point<high_resolution_clock> timepoint_s = high_resolution_clock::now();
    solveSeq(rows, cols, iters, td, h, sleep, matrix);
    time_point<high_resolution_clock> timepoint_e = high_resolution_clock::now();

    cout << "----- SEQUENTIAL -----" << endl << flush;
    printMatrix(rows, cols, matrix);

    long duration = duration_cast<microseconds>(timepoint_e - timepoint_s).count();

    deallocateMatrix(rows, matrix);
    return duration;
}

long parallel(int rows, int cols, int iters, double td, double h, int sleep, int rank, double ** matrix, int procCount)
{

    time_point<high_resolution_clock> timepoint_s, timepoint_e;
    timepoint_s = high_resolution_clock::now();

    bool toTransposeBack = false;

    // Transpose matrix if cols > rows to fit parallel solver per row
    if (cols > rows){
        matrix = transposeMatrix(matrix, rows, cols);
        int temp = rows;
        rows = cols;
        cols = temp; 
        toTransposeBack = true;
    }

    // Compute number of rows per thread
    int * threadRows = new int[procCount];
    int * displacements = new int[procCount];

    distributeRows(procCount, rows, cols, threadRows, displacements);

    double * bufferReceive = new double[cols * rows];
    double * bufferSend = new double[cols * threadRows[rank]];
    double ** matrixLoc = allocateMatrix(threadRows[rank], cols);

    // Fill the local matrix
    for (int i = 0; i < threadRows[rank]; i++){
        memcpy(matrixLoc[i], matrix[i + displacements[rank]], cols * sizeof(double));
    }
   
    solvePar(threadRows[rank], cols, iters, td, h, sleep, matrixLoc, rank, procCount - 1);

    // Build buffer to send to root
    for (int i = 0; i < threadRows[rank]; i++){
        memcpy(bufferSend + i * cols, matrixLoc[i], cols * sizeof(double));
    }

    int * recvCount = new int[procCount];
    int * disp = new int[procCount];

    for (int i = 0; i < procCount; i++){
        recvCount[i] = threadRows[i] * cols;
        disp[i] = displacements[i] * cols;

    }

    MPI_Gatherv(bufferSend, recvCount[rank], MPI_DOUBLE, bufferReceive, recvCount, disp, MPI_DOUBLE, 0, MPI_COMM_WORLD);  

    if (rank == 0)
    {

        for (int i = 0; i < rows; i++){
            memcpy(matrix[i], bufferReceive + i * cols, cols * sizeof(double));
        }

        if (toTransposeBack){
            matrix = transposeMatrix(matrix, rows, cols);
            int temp = rows;
            rows = cols;
            cols = temp;  
        }

        cout << "-----  PARALLEL  -----" << endl << flush;
        printMatrix(rows, cols, matrix);

    }

    timepoint_e = high_resolution_clock::now();
    long duration = duration_cast<microseconds>(timepoint_e - timepoint_s).count();

    delete[](recvCount);
    delete[](disp);
    delete[](threadRows);
    delete[](displacements);


    return duration;
}


void distributeRows(int procCount, int rows, int cols, int * threadRows, int * displacements){
    int rowsPerProc = rows / procCount;
    int remainingRows = rows % procCount;

    int offset = 0;
    for (int i = 0; i < procCount; i++) {
        // Fill threadRows
        if (i < remainingRows) {
            threadRows[i] = rowsPerProc + 1;
        } else {
            threadRows[i] = rowsPerProc;
        }

        // // Fill displacements
        displacements[i] = offset;
        offset += threadRows[i];
    }
}





