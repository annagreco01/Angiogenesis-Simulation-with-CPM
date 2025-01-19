#ifndef CPM_H
#define CPM_H
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <string.h>
#include <lapacke.h> //LAPACK per le matrici dense
#include <cs.h> //CSparse per le matrici sparse
#include "write_matrix.h"
#include "calc_time.h"

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
#define D 0.1 // Coefficiente di diffusione
#define DECAY_RATE 0.01
#define SECRETION_RATE 0.1
#define DIFF_DT 0.01

// Macro per accedere agli elementi della matrice in ordine column-major
// per poter memorizzare i dati in maniera compatibile con LAPACKE
#define A(k,l) A[(l)*n + (k)]

// Array per memorizzare il volume attuale di ogni cella. 
//Sostituisce completamente sum(sum(lattice == current_cell)); e sum(sum(lattice == neighbor_cell)). 
//Permette di evitare di calcolare ogni volta il volume di ogni cella
extern int cell_volumes[NUM_CELLS + 1];  // L'indice 0 contiene lo stato iniziale, le celle iniziano da 1

// Verifica se il codice è compilato in C o C++
#ifdef __cplusplus
extern "C" { // extern serve per evitare problemi con il linkage del C++, quindi specifica che il codice è in C 
#endif

// Prototipi delle funzioni

// Funzione costruire il sistema lineare A * x = b usando LAPACK
void linear_system(int n, double* A, double* b);
// Funzione risolvere il sistema lineare A * x = b usando SuiteSparse
void solve_linear_system(double** chemical, double** E_field_x, double** E_field_y, int size);
// Funzione per costruire il sistema lineare A * x = b usando SuiteSparse
void sparse_linear_system(cs* A, double* b, int n);
// Funzione per risolvere il sistema lineare A * x = b usando SuiteSparse
void solve_sparse_linear_system(double** chemical, double** E_field_x, double** E_field_y, int size);
// Prototipi delle funzioni per il calcolo delle variazioni di energie
double calculate_delta_adhesion(int** lattice, int x, int y, int nx, int ny, int size);
// calculate_delta_volume fa uso dell'array cell_volumes[] 
double calculate_delta_volume(int current_volume, int neighbor_volume, int target_volume);
double calculate_delta_surface(int** lattice, int x, int y, int nx, int ny, int size);
double calculate_delta_chemotaxis(double** chemical, int x, int y, int nx, int ny);
double calculate_delta_electric(int x, int y, int nx, int ny, double** E_field_x, double** E_field_y);
double calculate_delta_total_energy(int** lattice, double** chemical, double** E_field_x, double** E_field_y, int x, int y, int nx, int ny, int current_volume, int neighbor_volume, int target_volume, int size);

// Prototipo della funzione per trovare il valore massimo nel campo chimico
double find_max_chemical(double** chemical, int size);
// Prototipo della funzione per normalizzare il campo chimico
void normalize_chemical_field(double** chemical, int size);
void get_random_neighbor(int x, int y, int *nx, int *ny);
void initialize_cells(int** lattice);
double** allocate_matrix(int size);
int** allocate_int_matrix(int size);
void free_matrix(double** matrix, int size);
void free_int_matrix(int** matrix, int size);
void initialize_electric_field(double** E_field_x, double** E_field_y, int size);
void update_chemical(double** chemical, int** lattice, int size);

#ifdef __cplusplus
}
#endif // Fine del blocco per il linkage C++
#ifdef DEBUG
    #define DEBUG_PRINT(...)    printf(__VA_ARGS__)
#else
    #define DEBUG_PRINT(...)
#endif

#endif // CPM_H