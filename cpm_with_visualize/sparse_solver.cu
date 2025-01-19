#include "sparse_solver.cuh"
#include <stdio.h>

SparseMatrix* initializeSparseMatrix(int n, int estimatedNnz) {
    DEBUG_PRINT("Inizializzazione matrice sparsa %d x %d\n", n, n);
    SparseMatrix* matrix = new SparseMatrix();
    matrix->n = n;
    matrix->nnz = 0;
    
    DEBUG_PRINT("Allocazione memoria su CPU\n");
    // Allocazione memoria su CPU
    matrix->csrRowPtr = (int*)malloc((n + 1) * sizeof(int));
    
    // Se estimatedNnz è 0, allochiamo comunque un minimo di spazio
    int initialCapacity = (estimatedNnz > 0) ? estimatedNnz : 1;
    matrix->csrColInd = (int*)malloc(initialCapacity * sizeof(int));
    matrix->csrVal = (double*)malloc(initialCapacity * sizeof(double));
    
    // Inizializzazione dell'array dei puntatori alle righe
    for(int i = 0; i <= n; i++) {
        matrix->csrRowPtr[i] = 0;
    }
    
    DEBUG_PRINT("Inizializzazione cuSPARSE\n");
    // Inizializzazione cuSPARSE
    CHECK_CUSPARSE(cusparseCreate(&matrix->cusparseHandle));
    
    return matrix;
}

SparseMatrixCOO* createCOOMatrix(int n, int nnz) {
    SparseMatrixCOO* cooMatrix = (SparseMatrixCOO*)malloc(sizeof(SparseMatrixCOO));
    cooMatrix->n = n;
    cooMatrix->nnz = nnz;

    // Allocazione memoria su GPU
    cudaMalloc(&cooMatrix->d_rowInd, nnz * sizeof(int));
    cudaMalloc(&cooMatrix->d_colInd, nnz * sizeof(int));
    cudaMalloc(&cooMatrix->d_values, nnz * sizeof(double));

    return cooMatrix;
}

void printCOOMatrix(int nnz, int* d_rowInd, int* d_colInd, double* d_values) {
    // Alloca memoria sull'host
    int* h_rowInd = (int*)malloc(nnz * sizeof(int));
    int* h_colInd = (int*)malloc(nnz * sizeof(int));
    double* h_values = (double*)malloc(nnz * sizeof(double));

    // Copia i dati dalla GPU alla CPU
    cudaMemcpy(h_rowInd, d_rowInd, nnz * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_colInd, d_colInd, nnz * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_values, d_values, nnz * sizeof(double), cudaMemcpyDeviceToHost);

    // Stampa i valori della matrice COO
    printf("COO Format:\n");
    for (int i = 0; i < nnz; i++) {
        printf("Row: %d, Col: %d, Val: %f\n", h_rowInd[i], h_colInd[i], h_values[i]);
    }

    // Libera la memoria sull'host
    free(h_rowInd);
    free(h_colInd);
    free(h_values);
}

/*
SparseMatrix* initializeCOOMatrix(int size, int nnz) {
    SparseMatrix* A = (SparseMatrix*)malloc(sizeof(SparseMatrix));
    A->size = size;
    A->nnz = nnz;

    cudaMalloc(&A->row_indices, nnz * sizeof(int));
    cudaMalloc(&A->col_indices, nnz * sizeof(int));
    cudaMalloc(&A->values, nnz * sizeof(double));

    return A;
}
*/

void convertCOOtoCSR(SparseMatrixCOO* cooMatrix, SparseMatrixCSR* csrMatrix) {
    // Copia i valori e gli indici di colonna da COO a CSR
    cudaMemcpy(csrMatrix->d_csrColInd, cooMatrix->d_colInd, cooMatrix->nnz * sizeof(int), cudaMemcpyDeviceToDevice);
    cudaMemcpy(csrMatrix->d_csrVal, cooMatrix->d_values, cooMatrix->nnz * sizeof(double), cudaMemcpyDeviceToDevice);

    // Conversione degli indici di riga
    cusparseXcoo2csr(
        csrMatrix->cusparseHandle,
        cooMatrix->d_rowInd,        // Indici di riga in formato COO
        cooMatrix->nnz,             // Numero di non-zero
        cooMatrix->n,               // Numero di righe
        csrMatrix->d_csrRowPtr,     // Puntatori alle righe in formato CSR
        CUSPARSE_INDEX_BASE_ZERO    // Base degli indici (0-based)
    );
}

SparseMatrixCSR* createCSRMatrix(int n, int nnz) {
    SparseMatrixCSR* csrMatrix = (SparseMatrixCSR*)malloc(sizeof(SparseMatrixCSR));
    csrMatrix->n = n;
    csrMatrix->nnz = nnz;
    size_t freeMem, totalMem;

    // Allocazione memoria su GPU
    cudaMemGetInfo(&freeMem, &totalMem);
    DEBUG_PRINT("Sto per allocare csrMatrix->d_csrRowPtr. Memoria libera: %zu bytes, Memoria totale: %zu bytes\n", freeMem, totalMem);    
    cudaMalloc(&csrMatrix->d_csrRowPtr, (n + 1) * sizeof(int));
    cudaMemGetInfo(&freeMem, &totalMem);
    DEBUG_PRINT("Sto per allocare csrMatrix->d_csrColPtr. Memoria libera: %zu bytes, Memoria totale: %zu bytes\n", freeMem, totalMem);   
    cudaMalloc(&csrMatrix->d_csrColInd, nnz * sizeof(int));
    cudaMemGetInfo(&freeMem, &totalMem);
    DEBUG_PRINT("Sto per allocare csrMatrix->d_csrVal. Memoria libera: %zu bytes, Memoria totale: %zu bytes\n", freeMem, totalMem);   
    cudaMalloc(&csrMatrix->d_csrVal, nnz * sizeof(double));

    // Creazione handle cuSPARSE
    cudaMemGetInfo(&freeMem, &totalMem);
    DEBUG_PRINT("Sto per fare cusparseCreate(). Memoria libera: %zu bytes, Memoria totale: %zu bytes\n", freeMem, totalMem);   
    cusparseCreate(&csrMatrix->cusparseHandle);

    return csrMatrix;
}

void printCSRMatrix(int n, int nnz, int* d_csrRowPtr, int* d_csrColInd, double* d_csrVal) {
    // Alloca memoria sull'host per i dati CSR
    int* h_csrRowPtr = (int*)malloc((n + 1) * sizeof(int));
    int* h_csrColInd = (int*)malloc(nnz * sizeof(int));
    double* h_csrVal = (double*)malloc(nnz * sizeof(double));

    // Copia i dati dalla GPU alla CPU
    cudaMemcpy(h_csrRowPtr, d_csrRowPtr, (n + 1) * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_csrColInd, d_csrColInd, nnz * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_csrVal, d_csrVal, nnz * sizeof(double), cudaMemcpyDeviceToHost);

    // Stampa il vettore dei puntatori alle righe
    printf("CSR Row Pointers:\n");
    for (int i = 0; i <= n; i++) {
        printf("%d ", h_csrRowPtr[i]);
    }
    printf("\n");

    // Stampa il vettore degli indici delle colonne
    printf("CSR Column Indices:\n");
    for (int i = 0; i < nnz; i++) {
        printf("%d ", h_csrColInd[i]);
    }
    printf("\n");

    // Stampa il vettore dei valori non nulli
    printf("CSR Values:\n");
    for (int i = 0; i < nnz; i++) {
        printf("%f ", h_csrVal[i]);
    }
    printf("\n");

    // Stampa il formato matrice leggibile (opzionale)
    printf("CSR Matrix in human-readable format:\n");
    for (int i = 0; i < n; i++) {
        for (int j = h_csrRowPtr[i]; j < h_csrRowPtr[i + 1]; j++) {
            printf("Row %d, Col %d, Value %f\n", i, h_csrColInd[j], h_csrVal[j]);
        }
    }

    // Libera la memoria sull'host
    free(h_csrRowPtr);
    free(h_csrColInd);
    free(h_csrVal);
}


/*
void solveWithCSR(SparseMatrixCSR* csrMatrix, double* b, double* x) {
    // Parametri per cuSPARSE
    cusparseMatDescr_t descrA; // Descrittore della matrice
    double tol = 1e-10;        // Tolleranza per il metodo
    int reorder = 0;           // Non applicare riordinamento
    int singularity = -1;      // Indica se la matrice è singolare

    // Crea il descrittore della matrice
    cusparseCreateMatDescr(&descrA);
    cusparseSetMatType(descrA, CUSPARSE_MATRIX_TYPE_GENERAL); // Matrice generale
    cusparseSetMatIndexBase(descrA, CUSPARSE_INDEX_BASE_ZERO); // Indici 0-based

    // Risoluzione del sistema lineare con il metodo QR
    cusparseStatus_t status = cusparseDcsrlsvqr(
        csrMatrix->cusparseHandle, // Handle cuSPARSE
        csrMatrix->n,              // Dimensione della matrice
        csrMatrix->nnz,            // Numero di non-zero
        descrA,                    // Descrittore della matrice
        csrMatrix->d_csrVal,       // Valori CSR
        csrMatrix->d_csrRowPtr,    // Puntatori alle righe CSR
        csrMatrix->d_csrColInd,    // Indici di colonna CSR
        b,                         // Vettore del termine noto
        tol,                       // Tolleranza per il metodo
        reorder,                   // Riordinamento (0 = disabilitato)
        x,                         // Vettore risultato
        &singularity               // Indicatore di singolarità
    );

    // Controlla se la matrice è singolare
    if (singularity >= 0) {
        fprintf(stderr, "Errore: La matrice è singolare in posizione %d\n", singularity);
        exit(EXIT_FAILURE);
    }

    // Controlla lo stato di cuSPARSE
    if (status != CUSPARSE_STATUS_SUCCESS) {
        fprintf(stderr, "Errore nella risoluzione del sistema lineare con cusparseDcsrlsvqr\n");
        exit(EXIT_FAILURE);
    }

    // Libera risorse
    cusparseDestroyMatDescr(descrA);
}

void solveWithCSR(SparseMatrixCSR* csrMatrix, double* b, double* x) {
    double *d_b, *d_x;
    size_t bufferSize;
    void* buffer = nullptr;

    // Allocazione dei vettori b e x sulla GPU
    CHECK_CUDA(cudaMalloc((void**)&d_b, csrMatrix->n * sizeof(double)));
    CHECK_CUDA(cudaMalloc((void**)&d_x, csrMatrix->n * sizeof(double)));

    // Copia del vettore b dalla CPU alla GPU
    CHECK_CUDA(cudaMemcpy(d_b, b, csrMatrix->n * sizeof(double), cudaMemcpyHostToDevice));

    // Creazione di descrittori per matrice CSR e vettori
    cusparseSpMatDescr_t matA;
    cusparseDnVecDescr_t vecX, vecB;

    CHECK_CUSPARSE(cusparseCreateCsr(
        &matA, csrMatrix->n, csrMatrix->n, csrMatrix->nnz,
        csrMatrix->d_csrRowPtr, csrMatrix->d_csrColInd, csrMatrix->d_csrVal,
        CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
        CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F)); // CSR con double

    CHECK_CUSPARSE(cusparseCreateDnVec(&vecX, csrMatrix->n, d_x, CUDA_R_64F));
    CHECK_CUSPARSE(cusparseCreateDnVec(&vecB, csrMatrix->n, d_b, CUDA_R_64F));

    // Parametri del prodotto matrice-vettore
    double alpha = 1.0;
    double beta = 0.0;

    // Calcolo della dimensione del buffer necessario
    CHECK_CUSPARSE(cusparseSpMV_bufferSize(
        csrMatrix->cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE,
        &alpha, matA, vecB, &beta, vecX, CUDA_R_64F,
        CUSPARSE_SPMV_ALG_DEFAULT, &bufferSize));

    // Allocazione del buffer
    CHECK_CUDA(cudaMalloc(&buffer, bufferSize));

    // Prodotto matrice-vettore CSR
    CHECK_CUSPARSE(cusparseSpMV(
        csrMatrix->cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE,
        &alpha, matA, vecB, &beta, vecX, CUDA_R_64F,
        CUSPARSE_SPMV_ALG_DEFAULT, buffer));

    // Copia del risultato x dalla GPU alla CPU
    CHECK_CUDA(cudaMemcpy(x, d_x, csrMatrix->n * sizeof(double), cudaMemcpyDeviceToHost));

    // Liberazione delle risorse
    CHECK_CUSPARSE(cusparseDestroySpMat(matA));
    CHECK_CUSPARSE(cusparseDestroyDnVec(vecX));
    CHECK_CUSPARSE(cusparseDestroyDnVec(vecB));
    cudaFree(buffer);
    cudaFree(d_b);
    cudaFree(d_x);
}
*/

void solveWithCSR(SparseMatrixCSR* csrMatrix, double* b, double* x) {
    double *d_b, *d_x;
    size_t bufferSize;
    void* buffer = nullptr;

    // Allocazione dei vettori b e x in memoria unificata
    CHECK_CUDA(cudaMalloc(&d_b, csrMatrix->n * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&d_x, csrMatrix->n * sizeof(double)));

    //CHECK_CUDA(cudaMallocManaged(&d_b, csrMatrix->n * sizeof(double)));
    //CHECK_CUDA(cudaMallocManaged(&d_x, csrMatrix->n * sizeof(double)));

    // Popolamento del vettore b (si può popolare direttamente perché è in memoria unificata)
    CHECK_CUDA(cudaMemcpy(d_b, b, csrMatrix->n * sizeof(double), cudaMemcpyHostToDevice));

    // Creazione dei descrittori per matrice CSR e vettori
    cusparseSpMatDescr_t matA;
    cusparseDnVecDescr_t vecX, vecB;

    CHECK_CUSPARSE(cusparseCreateCsr(
        &matA, csrMatrix->n, csrMatrix->n, csrMatrix->nnz,
        csrMatrix->d_csrRowPtr, csrMatrix->d_csrColInd, csrMatrix->d_csrVal,
        CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
        CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F)); // CSR con double

    CHECK_CUSPARSE(cusparseCreateDnVec(&vecX, csrMatrix->n, d_x, CUDA_R_64F));
    CHECK_CUSPARSE(cusparseCreateDnVec(&vecB, csrMatrix->n, d_b, CUDA_R_64F));

    // Parametri del prodotto matrice-vettore
    double alpha = 1.0;
    double beta = 0.0;

    // Calcolo della dimensione del buffer necessario
    CHECK_CUSPARSE(cusparseSpMV_bufferSize(
        csrMatrix->cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE,
        &alpha, matA, vecB, &beta, vecX, CUDA_R_64F,
        CUSPARSE_SPMV_ALG_DEFAULT, &bufferSize));

    // Sincronizza la GPU per assicurarsi che il prodotto matrice-vettore sia completato
    CHECK_CUDA(cudaDeviceSynchronize());

    // Allocazione del buffer
    CHECK_CUDA(cudaMalloc(&buffer, bufferSize));

    // Prodotto matrice-vettore CSR
    CHECK_CUSPARSE(cusparseSpMV(
        csrMatrix->cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE,
        &alpha, matA, vecB, &beta, vecX, CUDA_R_64F,
        CUSPARSE_SPMV_ALG_DEFAULT, buffer));

    // Il vettore x ora è direttamente accessibile senza bisogno di una copia esplicita
    memcpy(x, d_x, csrMatrix->n * sizeof(double));

    // Liberazione delle risorse
    CHECK_CUSPARSE(cusparseDestroySpMat(matA));
    CHECK_CUSPARSE(cusparseDestroyDnVec(vecX));
    CHECK_CUSPARSE(cusparseDestroyDnVec(vecB));
    cudaFree(buffer);
    cudaFree(d_b);
    cudaFree(d_x);
}

/*
__global__ void populateVectorB(double* d_b, double** chemical, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x; // Indice globale
    if (idx >= size * size) return;

    int i = idx / size; // Coordinata riga
    int j = idx % size; // Coordinata colonna

    d_b[idx] = chemical[i][j]; // Popola b con il valore chimico
}
*/
__global__ void populateVectorB(double* d_b, double* chemical, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x; // Indice globale
    if (idx >= size * size) return;

    d_b[idx] = chemical[idx]; // Popola b con il valore chimico
}

/*
__global__ void populateVectorB(double* d_b, double* d_chemical, size_t pitch, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;

    if (idx < size && idy < size) {
        // Ottieni il puntatore all'inizio della riga
        double* row = (double*)((char*)d_chemical + idy * pitch);
        d_b[idy * size + idx] = row[idx];
    }
}
*/

/*
__global__ void populateSparseMatrix(
    int* d_rowInd, int* d_colInd, double* d_values, double* d_b, double** chemical, int size) 
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x; // Indice globale del thread
    int n = size * size; // Dimensione totale della griglia

    if (idx >= n) return; // Fuori dai limiti

    int i = idx / size; // Coordinata riga
    int j = idx % size; // Coordinata colonna

    // Calcola il valore del vettore b in base al campo chimico e al modello fisico
    //d_b[idx] = chemical[i][j]; // Popola b con il valore chimico
    
    // Posizione di base per l'elemento corrente (massimo 5 valori per nodo)
    int basePos = idx * 5;

    // Diagonale principale
    d_rowInd[basePos] = idx;
    d_colInd[basePos] = idx;
    d_values[basePos] = 4.0;

    int pos = 1; // Contatore per gli elementi non nulli (escluso il diagonale)

    // Valori delle connessioni adiacenti (vicini)
    if (i > 0) { // Sopra
        d_rowInd[basePos + pos] = idx;
        d_colInd[basePos + pos] = idx - size;
        d_values[basePos + pos] = -1.0;
        pos++;
    }
    if (i < size - 1) { // Sotto
        d_rowInd[basePos + pos] = idx;
        d_colInd[basePos + pos] = idx + size;
        d_values[basePos + pos] = -1.0;
        pos++;
    }
    if (j > 0) { // Sinistra
        d_rowInd[basePos + pos] = idx;
        d_colInd[basePos + pos] = idx - 1;
        d_values[basePos + pos] = -1.0;
        pos++;
    }
    if (j < size - 1) { // Destra
        d_rowInd[basePos + pos] = idx;
        d_colInd[basePos + pos] = idx + 1;
        d_values[basePos + pos] = -1.0;
        pos++;
    }
}
*/

/*
__global__ void populateSparseMatrix(int* d_rowInd, int* d_colInd, double* d_values, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x; // Indice globale del thread
    int n = size * size; // Dimensione totale della griglia

    if (idx >= n) return; // Fuori dai limiti

// Per Row-Major-Order
//  int i = idx / size; // Coordinata riga
//  int j = idx % size; // Coordinata colonna
// Per Column-Major-Order
    int i = idx % size; // Coordinata riga
    int j = idx / size; // Coordinata colonna

    // Popolamento della diagonale principale
    d_rowInd[idx * 5] = idx; // moltiplico per 5 perché ogni nodo ha al massimo 5 connessioni
    d_colInd[idx * 5] = idx;
    d_values[idx * 5] = 4.0;

    int pos = 1;

    // Condizioni per i vicini
    if (i > 0) { // Sopra
        d_rowInd[idx * 5 + pos] = idx;
        d_colInd[idx * 5 + pos] = idx - size;
        d_values[idx * 5 + pos] = -1.0;
        pos++;
    }
    if (i < size - 1) { // Sotto
        d_rowInd[idx * 5 + pos] = idx;
        d_colInd[idx * 5 + pos] = idx + size;
        d_values[idx * 5 + pos] = -1.0;
        pos++;
    }
    if (j > 0) { // Sinistra
        d_rowInd[idx * 5 + pos] = idx;
        d_colInd[idx * 5 + pos] = idx - 1;
        d_values[idx * 5 + pos] = -1.0;
        pos++;
    }
    if (j < size - 1) { // Destra
        d_rowInd[idx * 5 + pos] = idx;
        d_colInd[idx * 5 + pos] = idx + 1;
        d_values[idx * 5 + pos] = -1.0;
    }
}
*/

__global__ void populateSparseMatrix(int* d_rowInd, int* d_colInd, double* d_values, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x; // Indice globale del thread
    int n = size * size;

    if (idx >= n) return;

    // Per ottenere il column-major order, iteriamo prima sulle colonne e poi sulle righe.
    int j = idx / size; // Coordinata colonna
    int i = idx % size; // Coordinata riga


    // Verifica che gli indici siano all'interno dei limiti della matrice
    if (i < 0 || i >= size || j < 0 || j >= size) {
        return; // Se gli indici sono fuori dai limiti, non fare nulla
    }

    // Popolamento della diagonale principale
    d_rowInd[idx * 5] = i; // Row indice basato su i
    d_colInd[idx * 5] = j; // Col indice basato su j
    d_values[idx * 5] = 4.0; // Valore della diagonale principale

    int pos = 1;

    // Valori delle connessioni adiacenti (vicini)
    if (i > 0) { // Sopra
        d_rowInd[idx * 5 + pos] = i - 1;
        d_colInd[idx * 5 + pos] = j;
        d_values[idx * 5 + pos] = -1.0;
        pos++;
    }
    if (i < size - 1) { // Sotto
        d_rowInd[idx * 5 + pos] = i + 1;
        d_colInd[idx * 5 + pos] = j;
        d_values[idx * 5 + pos] = -1.0;
        pos++;
    }
    if (j > 0) { // Sinistra
        d_rowInd[idx * 5 + pos] = i;
        d_colInd[idx * 5 + pos] = j - 1;
        d_values[idx * 5 + pos] = -1.0;
        pos++;
    }
    if (j < size - 1) { // Destra
        d_rowInd[idx * 5 + pos] = i;
        d_colInd[idx * 5 + pos] = j + 1;
        d_values[idx * 5 + pos] = -1.0;
    }
}


void setElement(SparseMatrix* matrix, int row, int col, double value) {
    if (row < 0 || row >= matrix->n || col < 0 || col >= matrix->n) {
        printf("Indici non validi: (%d, %d)\n", row, col);
        return;
    }
    
    DEBUG_PRINT("Inserimento elemento (%d, %d) = %f\n", row, col, value);
    // Cerchiamo se l'elemento esiste già
    int start = matrix->csrRowPtr[row];
    int end = matrix->csrRowPtr[row + 1];
    for(int i = start; i < end; i++) {
        if(matrix->csrColInd[i] == col) {
            matrix->csrVal[i] = value;
            return;
        }
    }
    
    // Se necessario, riallochiamo gli array con più spazio
    size_t currentCapacity = matrix->nnz;  // Assumiamo che la capacità sia uguale a nnz
    if (matrix->nnz >= currentCapacity) {
        size_t newCapacity = (currentCapacity == 0) ? 1 : currentCapacity * 2;

        DEBUG_PRINT("Riallocazione memoria: %d -> %d\n", currentCapacity, newCapacity);        
        // Riallochiamo gli array
        int* newColInd = (int*)realloc(matrix->csrColInd, newCapacity * sizeof(int));
        double* newVal = (double*)realloc(matrix->csrVal, newCapacity * sizeof(double));
        
        if (newColInd == nullptr || newVal == nullptr) {
            printf("Errore: impossibile allocare più memoria\n");
            return;
        }
        
        matrix->csrColInd = newColInd;
        matrix->csrVal = newVal;
    }
    
    DEBUG_PRINT("Inserimento nuovo elemento\n");
    // Spostiamo gli elementi esistenti
    for(int i = matrix->nnz; i > end; i--) {
        matrix->csrColInd[i] = matrix->csrColInd[i-1];
        matrix->csrVal[i] = matrix->csrVal[i-1];
    }
    
    // Inseriamo il nuovo elemento
    matrix->csrColInd[end] = col;
    matrix->csrVal[end] = value;
    matrix->nnz++;
    
    DEBUG_PRINT("Aggiornamento puntatori alle righe\n");
    // Aggiorniamo i puntatori alle righe
    for(int i = row + 1; i <= matrix->n; i++) {
        matrix->csrRowPtr[i]++;
    }
    DEBUG_PRINT("Fine inserimento\n");
}

void prepareForSolution(SparseMatrix* matrix) {
    CHECK_CUDA(cudaMalloc((void**)&matrix->d_csrRowPtr, (matrix->n + 1) * sizeof(int)));
    CHECK_CUDA(cudaMalloc((void**)&matrix->d_csrColInd, matrix->nnz * sizeof(int)));
    CHECK_CUDA(cudaMalloc((void**)&matrix->d_csrVal, matrix->nnz * sizeof(double))); // cambiato in double
    
    CHECK_CUDA(cudaMemcpy(matrix->d_csrRowPtr, matrix->csrRowPtr, 
                         (matrix->n + 1) * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(matrix->d_csrColInd, matrix->csrColInd, 
                         matrix->nnz * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(matrix->d_csrVal, matrix->csrVal, 
                         matrix->nnz * sizeof(double), cudaMemcpyHostToDevice)); // cambiato in double
}

void solveSparseSystem(SparseMatrix* matrix, double* b, double* x) { // cambiato in double
    double *d_b, *d_x;  // cambiato in double
    
    CHECK_CUDA(cudaMalloc((void**)&d_b, matrix->n * sizeof(double))); // cambiato in double
    CHECK_CUDA(cudaMalloc((void**)&d_x, matrix->n * sizeof(double))); // cambiato in double
    CHECK_CUDA(cudaMemcpy(d_b, b, matrix->n * sizeof(double), cudaMemcpyHostToDevice)); // cambiato in double
    
    cusparseSpMatDescr_t matA;
    cusparseDnVecDescr_t vecX, vecB;
    
    CHECK_CUSPARSE(cusparseCreateCsr(&matA, matrix->n, matrix->n, matrix->nnz,
                                    matrix->d_csrRowPtr, matrix->d_csrColInd, matrix->d_csrVal,
                                    CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                    CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F)); // cambiato in CUDA_R_64F
    
    CHECK_CUSPARSE(cusparseCreateDnVec(&vecX, matrix->n, d_x, CUDA_R_64F)); // cambiato in CUDA_R_64F
    CHECK_CUSPARSE(cusparseCreateDnVec(&vecB, matrix->n, d_b, CUDA_R_64F)); // cambiato in CUDA_R_64F
    
    double alpha = 1.0;  // cambiato in double
    double beta = 0.0;   // cambiato in double
    size_t bufferSize;
    void* buffer = nullptr;
    
    CHECK_CUSPARSE(cusparseSpMV_bufferSize(
        matrix->cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE,
        &alpha, matA, vecB, &beta, vecX, CUDA_R_64F, // cambiato in CUDA_R_64F
        CUSPARSE_SPMV_ALG_DEFAULT, &bufferSize));
    
    CHECK_CUDA(cudaMalloc(&buffer, bufferSize));
    
    CHECK_CUSPARSE(cusparseSpMV(
        matrix->cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE,
        &alpha, matA, vecB, &beta, vecX, CUDA_R_64F, // cambiato in CUDA_R_64F
        CUSPARSE_SPMV_ALG_DEFAULT, buffer));
    
    CHECK_CUDA(cudaMemcpy(x, d_x, matrix->n * sizeof(double), cudaMemcpyDeviceToHost)); // cambiato in double
    
    CHECK_CUSPARSE(cusparseDestroySpMat(matA));
    CHECK_CUSPARSE(cusparseDestroyDnVec(vecX));
    CHECK_CUSPARSE(cusparseDestroyDnVec(vecB));
    cudaFree(buffer);
    cudaFree(d_b);
    cudaFree(d_x);
}

void freeSparseMatrix(SparseMatrix* matrix) {
    if (matrix == nullptr) return;
    
    free(matrix->csrRowPtr);
    free(matrix->csrColInd);
    free(matrix->csrVal);
    
    cudaFree(matrix->d_csrRowPtr);
    cudaFree(matrix->d_csrColInd);
    cudaFree(matrix->d_csrVal);
    
    cusparseDestroy(matrix->cusparseHandle);
    
    delete matrix;
}