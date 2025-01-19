#ifndef SPARSE_SOLVER_H
#define SPARSE_SOLVER_H

#include <cuda_runtime.h>
#include <cusparse.h>

#undef D

//#include <cusolverDn.h>
#include <cusolverSp.h>

// Funzione per controllare gli errori CUDA
inline void checkCudaError(cudaError_t error, const char *file, int line) {
    if (error != cudaSuccess) {
        printf("CUDA Error at %s:%d: %s\n", file, line, cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }
}

// Funzione per controllare gli errori cuSPARSE
inline void checkCusparseError(cusparseStatus_t status, const char *file, int line) {
    if (status != CUSPARSE_STATUS_SUCCESS) {
        printf("cuSPARSE Error at %s:%d: %d\n", file, line, status);
        exit(EXIT_FAILURE);
    }
}

// Funzione per controllare gli errori cuSOLVER
inline void checkCusolverError(cusolverStatus_t status, const char *file, int line) {
    if (status != CUSOLVER_STATUS_SUCCESS) {
        printf("cuSOLVER Error at %s:%d: %d\n", file, line, status);
        exit(EXIT_FAILURE);
    }
}

#define CHECK_CUDA(err) checkCudaError(err, __FILE__, __LINE__)
#define CHECK_CUSPARSE(err) checkCusparseError(err, __FILE__, __LINE__)
#define CHECK_CUSOLVER(err) checkCusolverError(err, __FILE__, __LINE__)

// Struttura per gestire una matrice sparsa in formato CSR
struct SparseMatrix {
    int n;              // Dimensione della matrice (n x n)
    int nnz;           // Numero di elementi non-zero
    int *csrRowPtr;    // Array dei puntatori alle righe (CPU)
    int *csrColInd;    // Array degli indici di colonna (CPU)
    double *csrVal;    // Array dei valori (CPU) - cambiato in double
    int *d_csrRowPtr;  // Array dei puntatori alle righe (GPU)
    int *d_csrColInd;  // Array degli indici di colonna (GPU)
    double *d_csrVal;  // Array dei valori (GPU) - cambiato in double
    cusparseHandle_t cusparseHandle;
};

// Struttura per gestire una matrice sparsa in formato COO: Coordinate list (triplet) format. Cioè, una lista di triplette (row, col, value)
struct SparseMatrixCOO {
    int n;              // Dimensione della matrice (n x n)
    int nnz;            // Numero di elementi non-zero
    int *d_rowInd;      // Array degli indici di riga (GPU)
    int *d_colInd;      // Array degli indici di colonna (GPU)
    double *d_values;   // Array dei valori (GPU)
};

// Struttura per gestire una matrice sparsa in formato CSR: Compressed Sparse Row format. Cioè, una lista di righe, una lista di colonne e una lista di valori
struct SparseMatrixCSR {
    int n;              // Dimensione della matrice (n x n)
    int nnz;            // Numero di elementi non-zero
    int *d_csrRowPtr;   // Array dei puntatori alle righe (GPU)
    int *d_csrColInd;   // Array degli indici di colonna (GPU)
    double *d_csrVal;   // Array dei valori (GPU)
    cusparseHandle_t cusparseHandle;  // Handle per cuSPARSE
};


// Funzioni pubbliche
SparseMatrix* initializeSparseMatrix(int n, int estimatedNnz);

SparseMatrixCOO* createCOOMatrix(int n, int nnz);
void printCOOMatrix(int nnz, int* d_rowInd, int* d_colInd, double* d_values);
SparseMatrixCSR* createCSRMatrix(int n, int nnz);
void printCSRMatrix(int n, int nnz, int* d_csrRowPtr, int* d_csrColInd, double* d_csrVal);

SparseMatrix* initializeCOOMatrix(int size, int nnz);
void convertCOOtoCSR(SparseMatrixCOO* cooMatrix, SparseMatrixCSR* csrMatrix);
void solveWithCSR(SparseMatrixCSR* csrMatrix, double* b, double* x);

void solveWithCSR(SparseMatrix* A, double* b, double* x, cusparseHandle_t handle);
//__global__ void populateVectorB(double* d_b, double* d_chemical, size_t pitch, int size);
__global__ void populateVectorB(double* d_b, double* chemical, int size);
//__global__ void populateVectorB(double* d_b, double** chemical, int size);

__global__ void populateSparseMatrix(int* d_rowInd, int* d_colInd, double* d_values, int size);
//__global__ void populateSparseMatrix(int* d_rowInd, int* d_colInd, double* d_values, double* d_b, double** chemical, int size);

void setElement(SparseMatrix* matrix, int row, int col, double value); // cambiato in double
void prepareForSolution(SparseMatrix* matrix);
void solveSparseSystem(SparseMatrix* matrix, double* b, double* x); // cambiato in double
void freeSparseMatrix(SparseMatrix* matrix);

#ifdef DEBUG
    #define DEBUG_PRINT(...)    printf(__VA_ARGS__)
#else
    #define DEBUG_PRINT(...)
#endif

#endif // SPARSE_SOLVER_H