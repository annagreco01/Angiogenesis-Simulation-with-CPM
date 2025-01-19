#include "cpm.h"

// Funzione per risolvere il sistema lineare A * x = b usando LAPACK
void linear_system(int n, double* A, double* b) {
    int lda = n;  // leading dimension of A (number of rows)
    //int ipiv[n];  // array for pivot indices
    lapack_int* ipiv = (lapack_int*)calloc(n, sizeof(lapack_int));
    int nrhs = 1; // number of right-hand sides (columns in b)
    int info;     // output status from LAPACK

    if (ipiv == NULL || A == NULL || b == NULL) {
        printf("Errore nell'allocazione di memoria.\n");
        return;
    }
    // Chiamata a LAPACK: dgesv risolve il sistema A * x = b
    info = LAPACKE_dgesv(LAPACK_COL_MAJOR, n, nrhs, A, lda, ipiv, b, lda);
    // Controlla se la soluzione è stata trovata
    if (info == 0) {
        printf("Sistema risolto con successo.\n");
    } else if (info < 0) {
        printf("Errore: il %d-esimo parametro ha un valore illegale.\n", -info);
    } else if (info > 0) {
        printf("Errore: %d la matrice è singolare e non ha soluzioni.\n", info);
    }
    free(ipiv);
}

/* 
   Funzione per risolvere direttamente il sistema lineare A * x = b usando SuiteSparse 
   per effettuare la fattorizzazione LU e risolvere il sistema. Fa uso di matrici sparse 
   in formato Compressed Column Storage (CCS), 
   cioè 3 array per memorizzare i valori non nulli, le righe e le colonne.
*/
void sparse_linear_system(cs* A, double* b, int n) {
    css* S;     // Struttura per la fattorizzazione simbolica
    csn* N;     // Struttura per la fattorizzazione numerica
    double* x = (double*)calloc(n, sizeof(double));  // Soluzione

    // Fattorizzazione simbolica
    S = cs_sqr(2, A, 0);  // Analisi simbolica per LU
    if (!S) {
        printf("Errore: fattorizzazione simbolica fallita.\n");
        free(x);
        return;
    }

    // Fattorizzazione numerica
    N = cs_lu(A, S, 1e-12);  // Fattorizzazione numerica LU
    if (!N) {
        printf("Errore: fattorizzazione numerica fallita. La matrice potrebbe essere singolare.\n");
        cs_sfree(S);
        free(x);
        return;
    }

    // Risolvi il sistema
    cs_ipvec(N->pinv, b, x, n);  // Applica la permutazione
    cs_lsolve(N->L, x);          // Risolvi L * y = P * b
    cs_usolve(N->U, x);          // Risolvi U * x = y
    cs_ipvec(S->q, x, b, n);     // Riordina il risultato finale in b

    // Libera memoria
    cs_nfree(N);
    cs_sfree(S);
    free(x);
}

void solve_linear_system(double** chemical, double** E_field_x, double** E_field_y, int size) {
    // Dimensione del sistema lineare
    int n = size * size;
    double* A = (double*)calloc(n * n, sizeof(double));  // Matrice A allocata con zero
    double* b = (double*)calloc(n, sizeof(double));      // Vettore b allocato con zero

    // Costruisci il sistema lineare
    int idx;
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            idx = i * size + j;
            A(idx, idx) = 4.0; // Accedo con la macro A(k,l)
            if (i > 0) {
                A(idx, idx - size) = -1.0;  
            }
            if (i < size - 1) {
                A(idx, idx + size) = -1.0;
            }
            if (j > 0) {
                A(idx, idx - 1) = -1.0;
            }
            if (j < size - 1) {
                A(idx, idx + 1) = -1.0;
            }
            b[idx] = chemical[i][j];
        }
    }

    // Risolvi il sistema lineare
    linear_system(n, A, b);

    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            E_field_x[i][j] = 0.0;
            E_field_y[i][j] = 0.0;
            if (i > 0 && i < size - 1) {
                E_field_x[i][j] = (b[(i + 1) * size + j] - b[(i-1) * size + j]) / 2.0;
            }
            if (j > 0 && j < size - 1) {
                E_field_y[i][j] = (b[i * size + j + 1] - b[i * size + j - 1]) / 2.0;
            }
        }
    }
    free(b);
    free(A);
}
/*
   Questa funzione si occupa di costruire un sistema lineare sparso A*x=b utilizzando SuiteSparse 
   e di risolverlo, delegando parte della soluzione alla funzione sparse_linear_system().
   A è una matrice sparsa costruita in base alle connessioni della griglia e b è il campo chimico. 
   Una volta risolto il sistema, calcola i campi elettrici E_field_x e E_field_y 
   utilizzando le differenze finite sui valori di b.
*/
void solve_sparse_linear_system(double** chemical, double** E_field_x, double** E_field_y, int size) {
    // Dimensione del sistema lineare
    int n = size * size;

    // Costruzione della matrice sparsa A in formato triplet
    cs* A = cs_spalloc(n, n, n * n, 1, 1);  // Matrice sparsa allocata con spazio per n*n (terzo parametro) elementi non nulli. Sarebbe meglio calcolare il numero di elementi non nulli in anticipo per ottimizzare
    double* b = (double*)calloc(n, sizeof(double));  // Vettore b allocato con zero

    if (!A || !b) {
        printf("Errore nell'allocazione della memoria per A o b.\n");
        return;
    }

    // stampa di chemical originale per verifica
/*    DEBUG_PRINT("Stampo chemical originale per verifica\n");
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            if(chemical[i][j] != 0.0)
                DEBUG_PRINT("%f ", chemical[i][j]);
        }
    }
*/
    // Costruisci il sistema lineare
    int idx;
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            idx = i * size + j;

            // Valori diagonali
            cs_entry(A, idx, idx, 4.0);
            // Valori delle connessioni adiacenti (per i vicini)
            if (i > 0) {
                cs_entry(A, idx, idx - size, -1.0);
            }
            if (i < size - 1) {
                cs_entry(A, idx, idx + size, -1.0);
            }
            if (j > 0) {
                cs_entry(A, idx, idx - 1, -1.0);
            }
            if (j < size - 1) {
                cs_entry(A, idx, idx + 1, -1.0);
            }

            // Costruisci il vettore b
            b[idx] = chemical[i][j];
        }
    }

    // Converti la matrice da formato triplet a formato compress-column
    cs* Acsc = cs_compress(A);
    cs_spfree(A);  // Libera la matrice triplet

    // Risolvi il sistema lineare
    sparse_linear_system(Acsc, b, n);

    // Calcola i campi E_field_x e E_field_y
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            E_field_x[i][j] = 0.0;
            E_field_y[i][j] = 0.0;
            if (i > 0 && i < size - 1) {
                E_field_x[i][j] = (b[(i + 1) * size + j] - b[(i - 1) * size + j]) / 2.0;
            }
            if (j > 0 && j < size - 1) {
                E_field_y[i][j] = (b[i * size + j + 1] - b[i * size + j - 1]) / 2.0;
            }
        }
    }

    // Libera memoria
    free(b);
    cs_spfree(Acsc);  // Libera la matrice compressa
}

// Funzione per calcolare il cambiamento di energia di adesione
double calculate_delta_adhesion(int** lattice, int x, int y, int nx, int ny, int size) {
    int current_cell = lattice[x][y];
    int neighbor_cell = lattice[nx][ny];

    double delta_H = 0.0;
    int lx = size;
    int ly = size;

    // Cicli per analizzare i vicini della cella corrente
    for (int i = -1; i <= 1; i++) {
        for (int j = -1; j <= 1; j++) {
            if (i == 0 && j == 0) {
                continue;  // Salta la cella corrente
            }

            // Coordinate dei vicini della cella corrente, usando condizioni periodiche
            int xx = (x + i + lx) % lx;
            int yy = (y + j + ly) % ly;

            // Se il vicino ha un ID diverso dalla cella corrente, diminuisci l'energia
            if (lattice[xx][yy] != current_cell) {
                delta_H -= 1.0;
            }

            // Aumenta l'energia se il vicino del candidato è diverso
            if (lattice[xx][yy] != neighbor_cell) {
                delta_H += 1.0;
            }
        }
    }

    return delta_H;
}

// Funzione per calcolare il cambiamento di energia di volume
double calculate_delta_volume(int current_volume, int neighbor_volume, int target_volume) {
    double delta_H = LAMBDA_V * (
        pow(neighbor_volume + 1 - target_volume, 2) +
        pow(current_volume - 1 - target_volume, 2) -
        pow(neighbor_volume - target_volume, 2) -
        pow(current_volume - target_volume, 2)
    );
    return delta_H;
}

// Funzione per calcolare il cambiamento di energia di superficie
double calculate_delta_surface(int** lattice, int x, int y, int nx, int ny, int size) {
    int current_cell = lattice[x][y];
    int neighbor_cell = lattice[nx][ny];

    double delta_S_current = 0;
    double delta_S_neighbor = 0;

    // Ciclo per analizzare i vicini della cella
    for (int i = -1; i <= 1; i++) {
        for (int j = -1; j <= 1; j++) {
            if (i == 0 && j == 0) continue;

            int xx = (x + i + size) % size;
            int yy = (y + j + size) % size;

            if (lattice[xx][yy] != current_cell) {
                delta_S_current--;
            }
            if (lattice[xx][yy] != neighbor_cell) {
                delta_S_neighbor++;
            }
        }
    }

    return LAMBDA_S * (delta_S_current + delta_S_neighbor);
}

// Funzione per calcolare il cambiamento di energia di chemiotassi
double calculate_delta_chemotaxis(double** chemical, int x, int y, int nx, int ny) {
    double delta_H = -LAMBDA_C * (chemical[nx][ny] - chemical[x][y]);
    return delta_H;
}

// Funzione per calcolare il cambiamento di energia del campo elettrico
double calculate_delta_electric(int x, int y, int nx, int ny, double** E_field_x, double** E_field_y) {
    int delta_x = nx - x;
    int delta_y = ny - y;
    double delta_H = -LAMBDA_E * (E_field_x[x][y] * delta_x + E_field_y[x][y] * delta_y);
    return delta_H;
}

// Funzione per calcolare l'energia totale
double calculate_delta_total_energy(int** lattice, double** chemical, double** E_field_x, double** E_field_y, int x, int y, int nx, int ny, int current_volume, int neighbor_volume, int target_volume, int size) {
    if (lattice[x][y] == lattice[nx][ny]) {
        return 0.0;  // Nessun cambiamento se le celle sono uguali
    }
    double delta_H_adhesion = calculate_delta_adhesion(lattice, x, y, nx, ny, size);
    double delta_H_volume = 0;
    double delta_H_surface = 0;
    // Questo test evita di calcolare delta_H_volume e delta_H_surface se non è necessario
    // Differisce dalla versione Matlab che lo faceva all'interno delle funzioni, quindi entrando in ogni funzione
    if (!(lattice[x][y] == 0 || lattice[nx][ny] == 0)) {
        delta_H_volume = calculate_delta_volume(current_volume, neighbor_volume, target_volume);
        delta_H_surface = calculate_delta_surface(lattice, x, y, nx, ny, size);
    }
    double delta_H_chemotaxis = calculate_delta_chemotaxis(chemical, x, y, nx, ny);
    double delta_H_electric = calculate_delta_electric(x, y, nx, ny, E_field_x, E_field_y);

    return delta_H_adhesion + delta_H_volume + delta_H_surface + delta_H_chemotaxis + delta_H_electric;
}

// Funzione per selezionare un vicino casuale
void get_random_neighbor(int x, int y, int *nx, int *ny) {
    int dir = rand() % 4;
    switch (dir) {
        case 0: *nx = (x + 1) % LATTICE_SIZE; *ny = y; break;
        case 1: *nx = (x - 1 + LATTICE_SIZE) % LATTICE_SIZE; *ny = y; break;
        case 2: *nx = x; *ny = (y + 1) % LATTICE_SIZE; break;
        case 3: *nx = x; *ny = (y - 1 + LATTICE_SIZE) % LATTICE_SIZE; break;
    }
}

// Allocazione e liberazione di una matrice
double** allocate_matrix(int size) {
    double** matrix = (double**) malloc(size * sizeof(double*));
    if (matrix == NULL) {
        fprintf(stderr, "Errore allocazione memoria per la matrice.\n");
        exit(EXIT_FAILURE);
    }
    for (int i = 0; i < size; i++) {
        matrix[i] = (double*) calloc(size, sizeof(double));
        if (matrix[i] == NULL) {
            fprintf(stderr, "Errore allocazione memoria per le righe della matrice.\n");
            for (int j = 0; j < i; j++) {
                free(matrix[j]);
            }
            free(matrix);
            exit(EXIT_FAILURE);
        }
    }
    return matrix;
}

void free_matrix(double** matrix, int size) {
    for (int i = 0; i < size; i++) {
        free(matrix[i]);
    }
    free(matrix);
}

// Inizializzazione del campo elettrico
void initialize_electric_field(double** E_field_x, double** E_field_y, int size) {
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            E_field_x[i][j] = 0.0;
            E_field_y[i][j] = 0.0;
        }
    }
}

// Funzione per trovare il valore massimo nella matrice chimica
double find_max_chemical(double** chemical, int size) {
    double max_value = chemical[0][0];
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            if (chemical[i][j] > max_value) {
                max_value = chemical[i][j];
            }
        }
    }
    return max_value;
}

// Funzione per normalizzare il campo chimico
void normalize_chemical_field(double** chemical, int size) {
    double max_value = find_max_chemical(chemical, size);
    
    // Assicurati che il massimo non sia zero per evitare divisione per zero
    if (max_value > 0) {
        for (int i = 0; i < size; i++) {
            for (int j = 0; j < size; j++) {
                chemical[i][j] /= max_value;
            }
        }
    }
}

// Funzione per inizializzare le celle e distribuire il volume
void initialize_cells(int** lattice) {
    for (int cell = 1; cell <= NUM_CELLS; cell++) {
        int volume_assigned = 0;

        // Assegna ogni cellula a siti casuali fino a raggiungere il volume target
        while (volume_assigned < TARGET_VOLUME) {
            int x = rand() % LATTICE_SIZE;
            int y = rand() % LATTICE_SIZE;

            // Verifica se il sito è vuoto
            if (lattice[x][y] == 0) {
                lattice[x][y] = cell;  // Assegna la cella a questo sito
                cell_volumes[cell]++;  // Aumenta il volume della cella
                volume_assigned++;
            }
        }
    }
}

int** allocate_int_matrix(int size) {
    int** matrix = (int**) malloc(size * sizeof(int*));
    if (matrix == NULL) {
        fprintf(stderr, "Errore allocazione memoria per la matrice.\n");
        exit(EXIT_FAILURE);
    }
    for (int i = 0; i < size; i++) {
        matrix[i] = (int*) calloc(size, sizeof(int));
        if (matrix[i] == NULL) {
            fprintf(stderr, "Errore allocazione memoria per le righe della matrice.\n");
            for (int j = 0; j < i; j++) {
                free(matrix[j]);
            }
            free(matrix);
            exit(EXIT_FAILURE);
        }
    }
    return matrix;
}

void free_int_matrix(int** matrix, int size) {
    for (int i = 0; i < size; i++) {
        free(matrix[i]);
    }
    free(matrix);
}
// Funzione per aggiornare il campo chimico
// Fa uso di puntatori a puntatori per evitare di copiare la matrice, quindi differisce dalla versione Matlab
// Per comodità si fa uso di due puntatori a matrice, uno per la matrice corrente e uno per la matrice successiva
void update_chemical(double** chemical, int** lattice, int size) {
    double** new_chemical = allocate_matrix(size);
    double** curr_chemical = new_chemical;
    double** prev_chemical = chemical;
    double** tmp;
    double laplacian;
    // Per usare i puntatori è importante che il numero di operazioni sia pari, 
    // Se fosse dispari alla fine fare una copia della matrice
    for (int t = 0; t < 10; t++) { 
        for (int x = 0; x < size; x++) {
            for (int y = 0; y < size; y++) {
                // Diffusion
                laplacian = -4 * prev_chemical[x][y];
                for (int i = -1; i <= 1; i++) {
                    for (int j = -1; j <= 1; j++) {
                        if (i == 0 && j == 0) continue;

                        int xx = (x + i + size) % size;
                        int yy = (y + j + size) % size;

                        laplacian += prev_chemical[xx][yy];
                    }
                }
                
                // Reaction-diffusion update
                curr_chemical[x][y] = prev_chemical[x][y] + DIFF_DT * ( D * laplacian -
                                        DECAY_RATE * prev_chemical[x][y] +
                                        SECRETION_RATE * (lattice[x][y] > 0 ? 1 : 0) );

            }
        }
        // swap
        tmp = curr_chemical;
        curr_chemical = prev_chemical;
        prev_chemical = tmp;
    }

    free_matrix(new_chemical, size);
}
