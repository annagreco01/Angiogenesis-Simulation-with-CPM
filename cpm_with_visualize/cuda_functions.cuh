#ifndef CUDA_FUNCTIONS_CUH
#define CUDA_FUNCTIONS_CUH

#include <cuda_runtime.h> // Funzioni di runtime di CUDA
#include <cuda.h> // API di CUDA per controllare la GPU

#undef T
#undef D

#include <cusolverSp.h> // libreria CUSOLVER per la risoluzione di sistemi lineari sparsi su GPU
#include <cusolverDn.h> // libreria CUSOLVER per la risoluzione di sistemi lineari densi su GPU
#include <cublas_v2.h> // libreria CUBLAS per le operazioni BLAS su GPU
#include <cusparse.h> // libreria CUSPARSE per le operazioni sparse su GPU
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <string.h>
#include <lapacke.h> //LAPACK per le matrici dense
#include <cs.h> //CSparse per le matrici sparse
#include "write_matrix.h"
#include "calc_time.h"
#include "cpm.h"
/*
// Definizione dei parametri principali
#define LATTICE_SIZE 200
#define NUM_CELLS 10
#define TARGET_VOLUME 500
#define NUM_MCS 100
#define T 20.0  // Temperatura per l'algoritmo di Metropolis

// Parametri CPM
#define LAMBDA_V 50
#define LAMBDA_S 0.5
#define LAMBDA_C 100
#define LAMBDA_E 50

// Parametri del campo chimico
#define D 0.1
#define DECAY_RATE 0.01
#define SECRETION_RATE 0.1
#define DIFF_DT 0.01

// Macro per accedere agli elementi della matrice in ordine column-major
// per poter memorizzare i dati in maniera compatibile con LAPACKE
#define A(k,l) A[(l)*n + (k)]

// Array per memorizzare il volume attuale di ogni cella. 
//Sostituisce completamente sum(sum(lattice == current_cell)); e sum(sum(lattice == neighbor_cell)). 
//Permette di evitare di calcolare ogni volta il volume di ogni cella
extern int cell_volumes[NUM_CELLS + 1];  // L'indice 0 contiene lo stato iniziale, le celle iniziano da 1. Siccome è definita in cuda_functions.cu, va dichiarata come extern

*/

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

 // Prototipi delle funzioni

// Funzione per risolvere il sistema lineare A * x = b usando LAPACK
__global__ void build_linear_system_kernel(double* A, double* b, double* chemical, int size);
void solve_linear_system_cuda(double** chemical, double** E_field_x, double** E_field_y, int size);
//-------------------------------------------------------------------------------------------------
__global__ void compute_fields(const double* b, double* E_field_x, double* E_field_y, int size);
void solve_linear_system_cuSOLVER(double** chemical, double** E_field_x, double** E_field_y, int size);

void sparse_linear_system_cuda(cs* A, double* b, int n);
void solve_sparse_linear_system_cuda(double** chemical, double** E_field_x, double** E_field_y, int size);
// Funzione per risolvere il sistema lineare A * x = b usando SuiteSparse
//__global__ void build_sparse_linear_system_kernel(int* row_indices, int* col_indices, double* values, double* b, double* chemical, int size);
//void solve_sparse_linear_system_cuda(double** chemical, double** E_field_x, double** E_field_y, int size);

/*

// Prototipi delle funzioni per il calcolo delle variazioni di energie
__global__ void calculate_delta_adhesion_kernel(int* d_lattice, double* d_delta_H, int nx, int ny, int size);
void calculate_delta_adhesion_cuda(int* h_lattice, double* h_delta_H, int nx, int ny, int size);
// calculate_delta_volume fa uso dell'array cell_volumes[] 
__global__ void calculate_delta_volume_kernel(double* d_delta_H, int* d_current_volume, int* d_neighbor_volume, int target_volume, int num_elements);
void calculate_delta_volume_cuda(int* h_current_volume, int* h_neighbor_volume, double* h_delta_H, int target_volume, int num_elements);

__global__ void calculate_delta_surface_kernel(int* d_lattice, double* d_delta_S, int size, int nx, int ny);
void calculate_delta_surface_cuda(int* h_lattice, double* h_delta_S, int size, int nx, int ny);

__global__ void calculate_delta_chemotaxis_kernel(double* d_chemical, double* d_delta_H, int size);
void calculate_delta_chemotaxis_cuda(double* h_chemical, double* h_delta_H, int size);

__global__ void calculate_delta_electric_kernel(int* d_x, int* d_y, int* d_nx, int* d_ny, double* d_E_field_x, double* d_E_field_y, double* d_delta_H, int size);
void calculate_delta_electric_cuda(int* h_x, int* h_y, int* h_nx, int* h_ny, double** E_field_x, double** E_field_y, double* h_delta_H, int num_elements, int size);

__global__ void calculate_delta_total_energy_kernel(int* d_lattice, double* d_chemical, double* d_E_field_x, double* d_E_field_y, int* d_x, int* d_y, int* d_nx, int* d_ny, int* d_current_volume, int* d_neighbor_volume, double* d_delta_H_total, double* d_delta_H_adhesion, double* d_delta_H_volume, double* d_delta_H_surface, double* d_delta_H_chemotaxis, double* d_delta_H_electric, int target_volume, int size, int num_elements);
void calculate_delta_total_energy_cuda(int* h_lattice, double** chemical, double** E_field_x, double** E_field_y, int* h_x, int* h_y, int* h_nx, int* h_ny, int* h_current_volume, int* h_neighbor_volume, int target_volume, double* h_delta_H_total, int num_elements, int size);

// Prototipo della funzione per trovare il valore massimo nel campo chimico
__global__ void find_max_chemical_kernel(double* d_chemical, double* d_max, int size);
double find_max_chemical_cuda(double** chemical, int size);

// Prototipo della funzione per normalizzare il campo chimico
__global__ void normalize_chemical_field_kernel(double* d_chemical, double max_value, int size);
void normalize_chemical_field_cuda(double** chemical, int size);

__global__ void initialize_electric_field_kernel(double* d_E_field_x, double* d_E_field_y, int size);
void initialize_electric_field_cuda(double** E_field_x, double** E_field_y, int size);

__global__ void update_chemical_kernel(double* d_chemical, int* d_lattice, double* d_new_chemical, int size);
void update_chemical_cuda_shared(double* h_chemical, int* h_lattice, double* h_new_chemical, int size);

*/

/* ANCORA DA PARALLELIZZARE, già incluse in cpm.h
void get_random_neighbor(int x, int y, int *nx, int *ny);
void initialize_cells(int** lattice);
double** allocate_matrix(int size);
int** allocate_int_matrix(int size);
void free_matrix(double** matrix, int size);
void free_int_matrix(int** matrix, int size);
*/

#ifdef __cplusplus
}
#endif // __cplusplus

#ifdef DEBUG
    #define DEBUG_PRINT(...)    printf(__VA_ARGS__)
#else
    #define DEBUG_PRINT(...)
#endif

#endif // CUDA_FUNCTIONS_CUH
