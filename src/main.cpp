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

int main(int argc, char* argv[]) {
    // Arguments.
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
    // Calculate the number of rows for each process.
    // S'il n'est pas possible de distribuer les charges également entre les process, le root prend en chage l'excédent.
    // Exemple: 13 lignes, 5 colonnes sur 3 process
    // Rank 0 : 5 lignes
    // Rank 1 : 4 lignes
    // Rank 2 : 4 lignes

    time_point<high_resolution_clock> timepoint_s, timepoint_e;

    bool toTransposeBack = false;

    if (cols > rows){
        matrix = transposeMatrix(matrix, rows, cols);
        int temp = rows;
        rows = cols;
        cols = temp; 
        toTransposeBack = true;
    }

    int rowsRoot = rows / procCount + rows % procCount;
    int rowsProc = rows / procCount;

    double * bufferReceive = new double[cols * rows];
    double * bufferSend = new double[cols * rowsProc];

    double ** matrixLoc;
    
    timepoint_s = high_resolution_clock::now();

    if (rank == 0){

        solvePar(rowsRoot, cols, iters, td, h, sleep, matrix, rank, procCount - 1);
    }
    else{
        //Allocate a local matrix for each process
        matrixLoc = allocateMatrix(rowsProc, cols);
        
        // Fill the local matrix
        for (int i = 0; i < rowsProc; i++){
             memcpy(matrixLoc[i], matrix[i + rowsRoot + (rank - 1) * rowsProc], cols * sizeof(double));
        }
        solvePar(rowsProc, cols, iters, td, h, sleep, matrixLoc, rank, procCount - 1);
       
        //Construire un buffer 1D pour le gather
        //int k = 0;
        for (int i = 0; i < rowsProc; i++){
            memcpy(bufferSend + i * cols, matrixLoc[i], cols * sizeof(double));
        }
    }

   
    // Le root envoie des 0 à lui-meme. Seulement les valeurs des autres threads l'interesse. 
    MPI_Gather(bufferSend, rowsProc * cols, MPI_DOUBLE, bufferReceive, rowsProc * cols, MPI_DOUBLE, 0, MPI_COMM_WORLD);  
    timepoint_e = high_resolution_clock::now();
    long duration = duration_cast<microseconds>(timepoint_e - timepoint_s).count();
    if (rank == 0)
    {
        //Remplir la matrix avec les valeurs recues des autres threads
        for (int i = 0; i < rows - rowsRoot; i++){
            memcpy(matrix[rowsRoot + i], bufferReceive + cols * (rowsProc + i), cols * sizeof(double));
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
 

  

    // // deallocateMatrix(rows, matrix);
    // deallocateMatrix(rows, matrixLoc);
    // delete[](bufferReceive);
    // delete[](bufferSend);

    return duration;
}



