#include "cpm_cuda_functions.cuh"
#include "sparse_solver.cuh"

#include <iostream>

// Funzione per calcolare il numero massimo di non zeri in una matrice sparsa per una griglia size x size
size_t calculateNNZ(int size) {
    // Controllo dimensione minima
    if (size < 2) {
        std::cerr << "La griglia deve avere almeno dimensione 2x2\n";
        return 0;
    }

    // Calcolo nodi interni
    int internalNodes = (size - 2) * (size - 2); // Nodi interni
    size_t nnz_internal = internalNodes * 5;    // Ogni nodo interno ha al massimo 5 non zeri

    // Calcolo nodi sui bordi (esclusi gli angoli)
    int edgeNodes = 4 * (size - 2);             // 4 bordi, escludendo gli angoli
    size_t nnz_edges = edgeNodes * 4;           // Ogni nodo sul bordo ha al massimo 4 non zeri

    // Calcolo nodi agli angoli
    int cornerNodes = 4;                        // Sempre 4 angoli
    size_t nnz_corners = cornerNodes * 3;       // Ogni angolo ha al massimo 3 non zeri

    // Somma totale dei non zeri
    size_t totalNNZ = nnz_internal + nnz_edges + nnz_corners;

    return totalNNZ;
}

// Funzione per risolvere il sistema lineare A * x = b usando LAPACK
void linear_system(int n, double* A, double* b) {
    DEBUG_PRINT("Risoluzione del sistema lineare con LAPACK...\n");
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

void solve_linear_system(double** chemical, double** E_field_x, double** E_field_y, int size) {
    // Dimensione del sistema lineare
    int n = size * size;
    double* A = (double*)calloc(n * n, sizeof(double));  // Matrice A allocata con zero
    double* b = (double*)calloc(n, sizeof(double));      // Vettore b allocato con zero

    // Costruisci il sistema lineare
    DEBUG_PRINT("Costruzione del sistema lineare...\n");
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

    DEBUG_PRINT("Calcolo dei campi elettrici...\n");
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
   Questa funzione si occupa di costruire un sistema lineare sparso A*x=b utilizzando sparse_solver.cu 
   e di risolverlo, delegando parte della soluzione alla funzione sparse_linear_system().
   A è una matrice sparsa costruita in base alle connessioni della griglia e b è il campo chimico. 
   Una volta risolto il sistema, calcola i campi elettrici E_field_x e E_field_y 
   utilizzando le differenze finite sui valori di b.

void solve_sparse_linear_system(double** chemical, double** E_field_x, double** E_field_y, int size) {
    DEBUG_PRINT("Risoluzione del sistema lineare con sparse_solver...\n");
    // Dimensione del sistema lineare
    int n = size * size;
    DEBUG_PRINT("Dimensione matrice A: %d x %d\n", n*n, n*n);
    size_t NNZ = calculateNNZ(n); // Calcola il numero di non zeri per passarlo alla funzione di inizializzazione

    // Costruzione della matrice sparsa A in formato triplet
    //cs* A = cs_spalloc(n, n, n * n, 1, 1);  // Matrice sparsa allocata con spazio per n*n (terzo parametro) elementi non nulli. Sarebbe meglio calcolare il numero di elementi non nulli in anticipo per ottimizzare
    //SparseMatrix* A = initializeSparseMatrix(n * n, 0); // non sappimo quanti elementi ci saranno --- capire se è possibile prevederlo: si può migliorare molto le prestazioni
    SparseMatrix* A = initializeSparseMatrix(n, NNZ); // inizializza la matrice sparsa con il numero di non zeri calcolato
    DEBUG_PRINT("Matrice sparsa inizializzata\n");
    double* b = (double*)calloc(n, sizeof(double));  // Vettore b allocato con zero
    double* x = (double*)calloc(n, sizeof(double));

    if (!A || !b || !x) {
        printf("Errore nell'allocazione della memoria per A, x o b.\n");
        return;
    }

    DEBUG_PRINT("Costruzione del sistema lineare: popolo la matrice sparsa...\n");
    // Costruisci il sistema lineare
    int idx;
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            idx = i * size + j;

            DEBUG_PRINT("Costruzione riga %d, colonna %d, indice %d\n", i, j, idx);
            // Valori diagonali
            //cs_entry(A, idx, idx, 4.0);
            setElement(A, idx, idx, 4.0);
            // Valori delle connessioni adiacenti (per i vicini)
            if (i > 0) {
                //cs_entry(A, idx, idx - size, -1.0);
                setElement(A, idx, idx - size, -1.0);
            }
            if (i < size - 1) {
                //cs_entry(A, idx, idx + size, -1.0);
                setElement(A, idx, idx + size, -1.0);
            }
            if (j > 0) {
                //cs_entry(A, idx, idx - 1, -1.0);
                setElement(A, idx, idx -1, -1.0);
            }
            if (j < size - 1) {
                //cs_entry(A, idx, idx + 1, -1.0);
                setElement(A, idx, idx + 1, -1.0);
            }

            // Costruisci il vettore b
            b[idx] = chemical[i][j];
        }
    }

    // Converti la matrice da formato triplet a formato compress-column
    //cs* Acsc = cs_compress(A);
    //cs_spfree(A);  // Libera la matrice triplet
    DEBUG_PRINT("Preparazione per la soluzione con CUDA...\n");
    prepareForSolution(A);


    // Risolvi il sistema lineare
    //sparse_linear_system(Acsc, b, n);
    DEBUG_PRINT("Risoluzione del sistema lineare...\n");
    solveSparseSystem(A, b, x);

    DEBUG_PRINT("Calcolo dei campi elettrici...\n");
    // Calcola i campi E_field_x e E_field_y
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            E_field_x[i][j] = 0.0;
            E_field_y[i][j] = 0.0;
            if (i > 0 && i < size - 1) {
                E_field_x[i][j] = (x[(i + 1) * size + j] - x[(i - 1) * size + j]) / 2.0;
            }
            if (j > 0 && j < size - 1) {
                E_field_y[i][j] = (x[i * size + j + 1] - x[i * size + j - 1]) / 2.0;
            }
        }
    }

    DEBUG_PRINT("Liberazione della memoria...\n");
    // Libera memoria
    free(x);
    free(b);
    freeSparseMatrix(A);
    //cs_spfree(Acsc);  // Libera la matrice compressa
}
//

void solve_sparse_linear_system(double** chemical, double** E_field_x, double** E_field_y, int size) {
    DEBUG_PRINT("Risoluzione del sistema lineare con sparse_solver...\n");

    // Dimensione del sistema lineare
    int n = size * size;
    DEBUG_PRINT("Dimensione matrice A: %d x %d\n", n, n);

    // Creo un array 1D temporaneo per il campo chimico per parallelizzare progressivamente il codice sequenziale 
    // (in questo modo le funzioni sequenziali ancora presenti nel codice potranno essere usate senza modifiche)
    // A parallelizzazione completa, chemical sarà definitivamente un array 1D per consentire una parallelizzazione migliore su GPU.
    double* tempChemical = (double*)malloc(size * size * sizeof(double));
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            tempChemical[i * size + j] = chemical[i][j];
        }
    }


    double* d_chemical; // Chemical sulla GPU
    size_t pitch;      // Dimensione della riga in byte

    // Allocazione della memoria con `cudaMallocPitch` per garantire l'allineamento della memoria
    CHECK_CUDA(cudaMallocPitch((void**)&d_chemical, &pitch, size * sizeof(double), size));
    // Copia di chemical sulla GPU
    for (int i = 0; i < size; i++) {
    CHECK_CUDA(cudaMemcpy2D((void*)((char*)d_chemical + i * pitch), pitch,
                            (void*)chemical[i], size * sizeof(double),
                            size * sizeof(double), 1, cudaMemcpyHostToDevice));
}


    // stampa di chemical per verifica
    DEBUG_PRINT("Stampo chemical per verifica\n");
    for (int i = 0; i < n; i++) {
        if(tempChemical[i] != 0.0)
            DEBUG_PRINT("%f ", tempChemical[i]);
    }
    // stampa di chemical originale per verifica
    DEBUG_PRINT("Stampo chemical originale per verifica\n");
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            if(chemical[i][j] != 0.0)
                DEBUG_PRINT("%f ", chemical[i][j]);
        }
    }

    // Calcola il numero di elementi non zero
    size_t nnz = calculateNNZ(size);
    DEBUG_PRINT("Numero massimo di non zeri stimato: %zu\n", nnz);

    size_t freeMem, totalMem;
    cudaMemGetInfo(&freeMem, &totalMem);
    DEBUG_PRINT("Memoria libera: %zu bytes, Memoria totale: %zu bytes\n", freeMem, totalMem);

    // Inizializza la matrice COO
    SparseMatrixCOO* cooMatrix = createCOOMatrix(n, nnz);
    DEBUG_PRINT("Matrice COO inizializzata\n");

    // Allocazione del vettore b e popolamento
    double* d_b;
    //double* h_b = (double*)malloc(n * sizeof(double)); // Vettore temporaneo sulla CPU
    double* h_b = (double*)calloc(n, sizeof(double));  // Vettore b allocato con zero

    // Vettore b su memoria unificata per CPU e GPU al fine di limitare le copie
    double* b;
    CHECK_CUDA(cudaMalloc(&d_b, n * sizeof(double)));


    cudaMemGetInfo(&freeMem, &totalMem);
    DEBUG_PRINT("Memoria libera: %zu bytes, Memoria totale: %zu bytes\n", freeMem, totalMem);
    cudaMalloc(&d_b, n * sizeof(double));

    // Popola la matrice COO sulla GPU
    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
    
    cudaMemGetInfo(&freeMem, &totalMem);
    DEBUG_PRINT("Memoria libera: %zu bytes, Memoria totale: %zu bytes\n", freeMem, totalMem);

    //populateVectorB<<<blocksPerGrid, threadsPerBlock>>>(d_b, chemical, size);
    //   DEBUG_PRINT("Vettore b popolato sulla GPU\n");
    // RICORDA di sostitiure tempChemical con chemical a parallellizazione completa
    __global__ void populateVectorB(double* d_b, double* d_chemical, size_t pitch, int size);
    //populateVectorB<<<blocksPerGrid, threadsPerBlock>>>(d_b, tempChemical, size);
    cudaDeviceSynchronize();
    DEBUG_PRINT("Vettore b popolato sulla GPU con memoria unificata\n");

    CHECK_CUDA(cudaMemcpy(h_b, d_b, n * sizeof(double), cudaMemcpyDeviceToHost)); // Copia il vettore b dalla GPU alla CPU
    // stampa del vettore b per verifica
    DEBUG_PRINT("Stampo vettore b per verifica\n");
    for (int i = 0; i < n; i++) {
        if(h_b[i] != 0.0)
            DEBUG_PRINT("%f ", h_b[i]);
    }

    cudaMemGetInfo(&freeMem, &totalMem);
    DEBUG_PRINT("Memoria libera: %zu bytes, Memoria totale: %zu bytes\n", freeMem, totalMem);

    /*populateSparseMatrix<<<blocksPerGrid, threadsPerBlock>>>(
    cooMatrix->d_rowInd, cooMatrix->d_colInd, cooMatrix->d_values, 
    d_b, chemical, size
    );
    
    populateSparseMatrix<<<blocksPerGrid, threadsPerBlock>>>(cooMatrix->d_rowInd, cooMatrix->d_colInd, cooMatrix->d_values, size);

    CHECK_CUDA(cudaMemcpy(cooMatrix->d_values, h_b, n * sizeof(double), cudaMemcpyHostToDevice)); // Copia il vettore b sulla GPU

    cudaDeviceSynchronize();
    DEBUG_PRINT("Matrice COO popolata sulla GPU\n");

    cudaMemGetInfo(&freeMem, &totalMem);
    DEBUG_PRINT("Memoria libera: %zu bytes, Memoria totale: %zu bytes\n", freeMem, totalMem);

    // Converti COO in CSR
    SparseMatrixCSR* csrMatrix = createCSRMatrix(n, nnz);
    convertCOOtoCSR(cooMatrix, csrMatrix);
    DEBUG_PRINT("Matrice convertita da COO a CSR\n");

    cudaMemGetInfo(&freeMem, &totalMem);
    DEBUG_PRINT("Memoria libera: %zu bytes, Memoria totale: %zu bytes\n", freeMem, totalMem);

    // Risolve il sistema lineare con la matrice CSR
    double* d_x;
    cudaMalloc(&d_x, n * sizeof(double)); // Vettore soluzione
    solveWithCSR(csrMatrix, d_b, d_x);
    DEBUG_PRINT("Sistema lineare risolto\n");

    // Copia la soluzione dalla GPU alla CPU
    double* h_x = (double*)malloc(n * sizeof(double));
    cudaMemcpy(h_x, d_x, n * sizeof(double), cudaMemcpyDeviceToHost);

    // Calcolo dei campi elettrici
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            E_field_x[i][j] = 0.0;
            E_field_y[i][j] = 0.0;
            if (i > 0 && i < size - 1) {
                E_field_x[i][j] = (h_x[(i + 1) * size + j] - h_x[(i - 1) * size + j]) / 2.0;
            }
            if (j > 0 && j < size - 1) {
                E_field_y[i][j] = (h_x[i * size + j + 1] - h_x[i * size + j - 1]) / 2.0;
            }
        }
    }
    DEBUG_PRINT("Campi elettrici calcolati\n");

    // Libera memoria
    cudaFree(d_b);
    cudaFree(d_x);
    cudaFree(cooMatrix->d_rowInd);
    cudaFree(cooMatrix->d_colInd);
    cudaFree(cooMatrix->d_values);
    free(cooMatrix);
    cudaFree(csrMatrix->d_csrRowPtr);
    cudaFree(csrMatrix->d_csrColInd);
    cudaFree(csrMatrix->d_csrVal);
    cusparseDestroy(csrMatrix->cusparseHandle);
    free(csrMatrix);
    free(h_x);

    free(tempChemical);

    DEBUG_PRINT("Memoria liberata\n");
}
*/

void solve_sparse_linear_system(double** chemical, double** E_field_x, double** E_field_y, int size) {
    DEBUG_PRINT("Risoluzione del sistema lineare con cuSolver...\n");
    size_t freeMem, totalMem;  // Variabili per monitorare la memoria

    // Dimensione del sistema lineare
    int n = size * size;
    DEBUG_PRINT("Dimensione matrice A: %d x %d\n", n, n);

    // Creazione di una matrice temporanea chemical (1D)
    double* tempChemical = (double*)malloc(n * sizeof(double));
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            tempChemical[i * size + j] = chemical[i][j]; //Questo sarebbe in Row-major order
            //tempChemical[j * size + i] = chemical[i][j]; // Popola tempChemical in column-major order  per l'uso con cuSolver
        }
    }

    // Calcola il numero di elementi non zero
    size_t nnz = calculateNNZ(size);
    DEBUG_PRINT("Numero massimo di non zeri stimato: %zu\n", nnz);

    // Inizializza la matrice COO
    SparseMatrixCOO* cooMatrix = createCOOMatrix(n, nnz);
    DEBUG_PRINT("Matrice COO inizializzata\n");

    // Popola la matrice COO sulla GPU
    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
    populateSparseMatrix<<<blocksPerGrid, threadsPerBlock>>>(cooMatrix->d_rowInd, cooMatrix->d_colInd, cooMatrix->d_values, size);
    cudaDeviceSynchronize();
    DEBUG_PRINT("Matrice COO popolata sulla GPU\n");

    // Converti COO in CSR
    SparseMatrixCSR* csrMatrix = createCSRMatrix(n, nnz);
    convertCOOtoCSR(cooMatrix, csrMatrix);
    DEBUG_PRINT("Matrice convertita da COO a CSR\n");

    if (csrMatrix->d_csrRowPtr == nullptr || csrMatrix->d_csrColInd == nullptr || csrMatrix->d_csrVal == nullptr) {
        printf("Errore: Puntatori della matrice CSR non validi dopo convertCOOtoCSR.\n");
        exit(EXIT_FAILURE);
    }

    // Preparazione del descrittore della matrice CSR e handle cuSolver
    cusolverSpHandle_t cusolverHandle;
    cusparseMatDescr_t descrA;

    // Creazione di handle cuSolver e descrittore della matrice
    CHECK_CUSOLVER(cusolverSpCreate(&cusolverHandle)); // Creazione dell'handle cuSolver per la risoluzione del sistema lineare
    CHECK_CUSPARSE(cusparseCreateMatDescr(&descrA));  // Creazione del descrittore della matrice per la risoluzione del sistema lineare
    cusparseSetMatType(descrA, CUSPARSE_MATRIX_TYPE_GENERAL); // Tipo di matrice 
    cusparseSetMatIndexBase(descrA, CUSPARSE_INDEX_BASE_ZERO); // Indice base

    // Allocazione del vettore b e popolamento
    double* d_b; // Vettore b sulla GPU
    CHECK_CUDA(cudaMalloc((void**)&d_b, n * sizeof(double)));

    cudaMemGetInfo(&freeMem, &totalMem);
    DEBUG_PRINT("Memoria libera: %zu bytes, Memoria totale: %zu bytes, Memoria richiesta: %zu\n", freeMem, totalMem, n * sizeof(double));
    CHECK_CUDA(cudaMemcpy(d_b, tempChemical, n * sizeof(double), cudaMemcpyHostToDevice)); // Copia il vettore b sulla GPU
    
    DEBUG_PRINT("Vettore b popolato sulla GPU\n");

    // Allocazione del vettore soluzione x sulla GPU
    double* d_x;
    CHECK_CUDA(cudaMalloc((void**)&d_x, n * sizeof(double)));

    // Risolve il sistema Ax = b con cuSolver
    double tol = 1e-12;
    int reorder = 0;
    int singularity = -1; // È inizializzato a -1 e se il sistema è singolare, verrà sovrascritto con l'indice della riga singolare, altrimenti rimarrà -1

    cudaMemGetInfo(&freeMem, &totalMem);
    DEBUG_PRINT("Memoria libera: %zu bytes, Memoria totale: %zu bytes\n", freeMem, totalMem);

    DEBUG_PRINT("Sincronizzazione Device fatta\n");
    DEBUG_PRINT("cusolverSpDcsrlsvlu parameters:\n");
    DEBUG_PRINT("handle: %p\n", (void*)cusolverHandle);
    DEBUG_PRINT("n: %d\n", n);
    DEBUG_PRINT("nnzA: %d\n", nnz);
    DEBUG_PRINT("descrA: %p\n", (void*)descrA);
    DEBUG_PRINT("csrValA (pointer): %p\n", (void*)csrMatrix->d_csrVal);
    DEBUG_PRINT("csrRowPtrA (pointer): %p\n", (void*)csrMatrix->d_csrRowPtr);
    DEBUG_PRINT("csrColIndA (pointer): %p\n", (void*)csrMatrix->d_csrColInd);
    DEBUG_PRINT("b (pointer): %p\n", (void*)d_b);
    DEBUG_PRINT("tol: %lf\n", tol);
    DEBUG_PRINT("reorder: %d\n", reorder);
    DEBUG_PRINT("x (pointer): %p\n", (void*)d_x);
    DEBUG_PRINT("singularity (value: %d)\n", singularity);

/* 
    Fattorizzazione LU cusolverSpDcsrlsvlu
    Fattorizzazione QR cusolverSpDcsrlsvqr
    Fattorizzazione Cholesky cusolverSpDcsrlsvchol

    CHECK_CUSOLVER(cusolverSpDcsrlsvluHost(
        cusolverHandle, n, nnz, descrA,
        csrMatrix->d_csrVal, csrMatrix->d_csrRowPtr, csrMatrix->d_csrColInd,
        d_b, tol, reorder, d_x, &singularity));
    
*/
    cudaDeviceSynchronize();
    CHECK_CUSOLVER(cusolverSpDcsrlsvqr(cusolverHandle, n, nnz, descrA, csrMatrix->d_csrVal, csrMatrix->d_csrRowPtr, csrMatrix->d_csrColInd, d_b, tol, reorder, d_x, &singularity));

    if (singularity >= 0) {
        printf("La matrice A è singolare in posizione %d\n", singularity);
    } else {
        DEBUG_PRINT("Sistema risolto correttamente.\n");
    }

    cudaMemGetInfo(&freeMem, &totalMem);
    DEBUG_PRINT("Memoria libera: %zu bytes, Memoria totale: %zu bytes\n", freeMem, totalMem);
    // Copia della soluzione dalla GPU alla CPU
    double* h_x = (double*)malloc(n * sizeof(double));
    CHECK_CUDA(cudaMemcpy(h_x, d_x, n * sizeof(double), cudaMemcpyDeviceToHost));

    // Calcolo dei campi elettrici
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            E_field_x[i][j] = 0.0;
            E_field_y[i][j] = 0.0;
            if (i > 0 && i < size - 1) {
                E_field_x[i][j] = (h_x[(i + 1) * size + j] - h_x[(i - 1) * size + j]) / 2.0;
            }
            if (j > 0 && j < size - 1) {
                E_field_y[i][j] = (h_x[i * size + j + 1] - h_x[i * size + j - 1]) / 2.0;
            }
        }
    }
    DEBUG_PRINT("Campi elettrici calcolati\n");

    // Liberazione della memoria
    cudaFree(d_b);
    cudaFree(d_x);
    cudaFree(cooMatrix->d_rowInd);
    cudaFree(cooMatrix->d_colInd);
    cudaFree(cooMatrix->d_values);
    free(cooMatrix);
    cudaFree(csrMatrix->d_csrRowPtr);
    cudaFree(csrMatrix->d_csrColInd);
    cudaFree(csrMatrix->d_csrVal);
    cusparseDestroyMatDescr(descrA);
    cusolverSpDestroy(cusolverHandle);
    free(csrMatrix);
    free(h_x);
    free(tempChemical);

    DEBUG_PRINT("Memoria liberata e sistema risolto con successo.\n");
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
                curr_chemical[x][y] = prev_chemical[x][y] + DIFF_DT * ( DIFFUSION * laplacian -
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
