#include "cuda_functions.cuh"

/* Macro per verificare errori */
#define CUDA_CHECK(call) \
    if ((call) != cudaSuccess) { \
        printf("CUDA error at %s:%d\n", __FILE__, __LINE__); \
        exit(EXIT_FAILURE); \
    }

#define CUSOLVER_CHECK(call) \
    if ((call) != CUSOLVER_STATUS_SUCCESS) { \
        printf("cuSolver error at %s:%d\n", __FILE__, __LINE__); \
        exit(EXIT_FAILURE); \
    }

#define CUSPARSE_CHECK(call) \
    if ((call) != CUSPARSE_STATUS_SUCCESS) { \
        printf("cuSPARSE error at %s:%d\n", __FILE__, __LINE__); \
        exit(EXIT_FAILURE); \
    }


// Funzione per costruire il sistema lineare A * x = b con CUDA
__global__ void build_linear_system_kernel(double* A, double* b, double* chemical, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= size * size) return;

    int i = idx / size;
    int j = idx % size;

    // Costruisci il sistema lineare
    A[idx * size * size + idx] = 4.0;

    if (i > 0) {
        A[idx * size * size + (idx - size)] = -1.0;
    }
    if (i < size - 1) {
        A[idx * size * size + (idx + size)] = -1.0;
    }
    if (j > 0) {
        A[idx * size * size + (idx - 1)] = -1.0;
    }
    if (j < size - 1) {
        A[idx * size * size + (idx + 1)] = -1.0;
    }

    b[idx] = chemical[idx];
}

// Funzione per risolvere il sistema lineare A * x = b con CUDA
void solve_linear_system_cuda(double** chemical, double** E_field_x, double** E_field_y, int size) {
    int n = size * size;

    // Alloca memoria sulla GPU per A, b, e chemical
    double* d_A;
    double* d_b;
    double* d_chemical;

    cudaMalloc((void**)&d_A, n * n * sizeof(double));
    cudaMalloc((void**)&d_b, n * sizeof(double));
    cudaMalloc((void**)&d_chemical, n * sizeof(double));

    // Copia chemical sulla GPU
    cudaMemcpy(d_chemical, chemical[0], n * sizeof(double), cudaMemcpyHostToDevice);

    // Configura il kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;

    // Lancia il kernel per costruire la matrice A e il vettore b
    build_linear_system_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_b, d_chemical, size);
    cudaDeviceSynchronize();

    // Risolvi il sistema lineare usando cuSOLVER
    cusolverDnHandle_t handle;
    cusolverDnCreate(&handle);

    int* d_info;  // Output status
    int* d_ipiv;  // Pivot array
    cudaMalloc((void**)&d_info, sizeof(int));
    cudaMalloc((void**)&d_ipiv, n * sizeof(int));

    // Allocazione della memoria per il buffer di lavoro
    int lwork = 0;  // Dimensione del buffer di lavoro
    double* d_work; // Puntatore al buffer di lavoro

    // Calcola la dimensione del buffer di lavoro necessario per la fattorizzazione LU
    cusolverDnDgetrf_bufferSize(handle, n, n, d_A, n, &lwork);
    cudaMalloc((void**)&d_work, lwork * sizeof(double));

    // Passo 1: Fattorizzazione LU della matrice A
    cusolverDnDgetrf(handle, n, n, d_A, n, d_work, d_ipiv, d_info);

    // Passo 2: Risolvi il sistema A * x = b usando la fattorizzazione LU
    cusolverDnDgetrs(handle, CUBLAS_OP_N, n, 1, d_A, n, d_ipiv, d_b, n, d_info);

    // Copia il risultato dalla GPU alla CPU
    cudaMemcpy(chemical[0], d_b, n * sizeof(double), cudaMemcpyDeviceToHost);

    // Calcola E_field_x e E_field_y sulla CPU
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            E_field_x[i][j] = 0.0;
            E_field_y[i][j] = 0.0;

            if (i > 0 && i < size - 1) {
                E_field_x[i][j] = (chemical[i + 1][j] - chemical[i - 1][j]) / 2.0;
            }

            if (j > 0 && j < size - 1) {
                E_field_y[i][j] = (chemical[i][j + 1] - chemical[i][j - 1]) / 2.0;
            }
        }
    }

    // Libera la memoria sulla GPU
    cudaFree(d_A);
    cudaFree(d_b);
    cudaFree(d_chemical);
    cudaFree(d_info);
    cudaFree(d_ipiv);
    cudaFree(d_work);
    cusolverDnDestroy(handle);
}

/* Kernel per calcolare i campi elettrici */
__global__ void compute_fields(const double* b, double* E_field_x, double* E_field_y, int size) {
    // Calcola le coordinate globali del thread nella griglia
    int i = blockIdx.y * blockDim.y + threadIdx.y;  // Riga
    int j = blockIdx.x * blockDim.x + threadIdx.x;  // Colonna

    // Condizione per evitare accessi non validi ai bordi
    if (i >= 1 && i < size - 1 && j >= 1 && j < size - 1) {
        int idx = i * size + j; // Indice lineare per la griglia 2D
        // Calcolo delle derivate finite per i campi
        E_field_x[idx] = (b[(i + 1) * size + j] - b[(i - 1) * size + j]) / 2.0;
        E_field_y[idx] = (b[i * size + j + 1] - b[i * size + j - 1]) / 2.0;
    }
}

/* Funzione per risolvere il sistema lineare con cuSOLVER */
void solve_linear_system_cuSOLVER(double** chemical, double** E_field_x, double** E_field_y, int size) {
    int n = size * size;
    int lda = n;

    // Allocazione di memoria sull'host
    double* A = (double*)calloc(n * n, sizeof(double));
    double* b = (double*)calloc(n, sizeof(double));
    double *d_A, *d_b, *d_E_field_x, *d_E_field_y;
    int *d_info, *d_pivot;

    // Creazione della matrice A e del vettore b
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            int idx = i * size + j;
            A[idx * n + idx] = 4.0;
            if (i > 0) A[idx * n + (idx - size)] = -1.0;
            if (i < size - 1) A[idx * n + (idx + size)] = -1.0;
            if (j > 0) A[idx * n + (idx - 1)] = -1.0;
            if (j < size - 1) A[idx * n + (idx + 1)] = -1.0;
            b[idx] = chemical[i][j];
        }
    }

    // Allocazione memoria sulla GPU
    CUDA_CHECK(cudaMalloc((void**)&d_A, n * n * sizeof(double)));
    CUDA_CHECK(cudaMalloc((void**)&d_b, n * sizeof(double)));
    CUDA_CHECK(cudaMalloc((void**)&d_E_field_x, n * sizeof(double)));
    CUDA_CHECK(cudaMalloc((void**)&d_E_field_y, n * sizeof(double)));
    CUDA_CHECK(cudaMalloc((void**)&d_info, sizeof(int)));
    CUDA_CHECK(cudaMalloc((void**)&d_pivot, n * sizeof(int)));

    // Copia A e b su GPU
    CUDA_CHECK(cudaMemcpy(d_A, A, n * n * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, b, n * sizeof(double), cudaMemcpyHostToDevice));

    // Risoluzione con cuSOLVER
    cusolverDnHandle_t handle;
    CUSOLVER_CHECK(cusolverDnCreate(&handle));

    int workspace_size = 0;
    double* d_workspace;
    CUSOLVER_CHECK(cusolverDnDgetrf_bufferSize(handle, n, n, d_A, lda, &workspace_size));
    CUDA_CHECK(cudaMalloc((void**)&d_workspace, workspace_size * sizeof(double)));

    CUSOLVER_CHECK(cusolverDnDgetrf(handle, n, n, d_A, lda, d_workspace, d_pivot, d_info));
    CUSOLVER_CHECK(cusolverDnDgetrs(handle, CUBLAS_OP_N, n, 1, d_A, lda, d_pivot, d_b, lda, d_info));

    // Copia risultato di b indietro sull'host
    CUDA_CHECK(cudaMemcpy(b, d_b, n * sizeof(double), cudaMemcpyDeviceToHost));

    // Calcolo dei campi elettrici su GPU
    dim3 threads(16, 16);
    dim3 blocks((size + threads.x - 1) / threads.x, (size + threads.y - 1) / threads.y);

    compute_fields<<<blocks, threads>>>(d_b, d_E_field_x, d_E_field_y, size);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copia dei campi elettrici sull'host
    CUDA_CHECK(cudaMemcpy(*E_field_x, d_E_field_x, n * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(*E_field_y, d_E_field_y, n * sizeof(double), cudaMemcpyDeviceToHost));

    // Cleanup
    cudaFree(d_A);
    cudaFree(d_b);
    cudaFree(d_E_field_x);
    cudaFree(d_E_field_y);
    cudaFree(d_info);
    cudaFree(d_pivot);
    cudaFree(d_workspace);
    cusolverDnDestroy(handle);

    free(A);
    free(b);
}



/* 
   Funzione per risolvere direttamente il sistema lineare A * x = b usando SuiteSparse 
   per effettuare la fattorizzazione LU e risolvere il sistema. Fa uso di matrici sparse 
   in formato Compressed Column Storage (CCS), 
   cioè 3 array per memorizzare i valori non nulli, le righe e le colonne.
*/

// Funzione per risolvere il sistema lineare A * x = b con cuSolver e cuSPARSE
void sparse_linear_system_cuda(cs* A, double* b, int n) {
    // Handle per cuSolver e cuSPARSE
    cusolverSpHandle_t cusolver_handle;
    cusparseMatDescr_t descrA;

    // Allocazione memoria su GPU
    double *d_A_values, *d_b, *d_x;
    int *d_A_rowPtr, *d_A_colInd;

    CUDA_CHECK(cudaMalloc((void**)&d_A_values, A->nzmax * sizeof(double)));
    CUDA_CHECK(cudaMalloc((void**)&d_A_rowPtr, (n + 1) * sizeof(int)));
    CUDA_CHECK(cudaMalloc((void**)&d_A_colInd, A->nzmax * sizeof(int)));
    CUDA_CHECK(cudaMalloc((void**)&d_b, n * sizeof(double)));
    CUDA_CHECK(cudaMalloc((void**)&d_x, n * sizeof(double)));

    // Copia dati dalla CPU alla GPU
    CUDA_CHECK(cudaMemcpy(d_A_values, A->x, A->nzmax * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_A_rowPtr, A->p, (n + 1) * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_A_colInd, A->i, A->nzmax * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, b, n * sizeof(double), cudaMemcpyHostToDevice));

    // Creazione degli handle cuSolver/cuSPARSE
    CUSOLVER_CHECK(cusolverSpCreate(&cusolver_handle));
    CUSPARSE_CHECK(cusparseCreateMatDescr(&descrA));
    CUSPARSE_CHECK(cusparseSetMatType(descrA, CUSPARSE_MATRIX_TYPE_GENERAL));
    CUSPARSE_CHECK(cusparseSetMatIndexBase(descrA, CUSPARSE_INDEX_BASE_ZERO));

    // Risoluzione del sistema lineare
    int singularity;
    CUSOLVER_CHECK(cusolverSpDcsrlsvluHost(
        cusolver_handle, n, A->nzmax,
        descrA, d_A_values, d_A_rowPtr, d_A_colInd,
        d_b, 1e-12, 0, d_x, &singularity
    ));

    if (singularity >= 0) {
        printf("La matrice è singolare: colonna %d\n", singularity);
    }

    // Copia soluzione su CPU
    CUDA_CHECK(cudaMemcpy(b, d_x, n * sizeof(double), cudaMemcpyDeviceToHost));

    // Cleanup
    cudaFree(d_A_values);
    cudaFree(d_A_rowPtr);
    cudaFree(d_A_colInd);
    cudaFree(d_b);
    cudaFree(d_x);
    cusolverSpDestroy(cusolver_handle);
    cusparseDestroyMatDescr(descrA);
}

// Funzione per risolvere il sistema lineare A * x = b con cuSolver e cuSPARSE
void solve_sparse_linear_system_cuda(double** chemical, double** E_field_x, double** E_field_y, int size) {
    int n = size * size;

    // Creazione della matrice sparsa
    cs* A = cs_spalloc(n, n, n * n, 1, 1);
    double* b = (double*)calloc(n, sizeof(double));

    // Costruzione della matrice e del vettore b
    int idx;
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            idx = i * size + j;

            // Valori diagonali
            cs_entry(A, idx, idx, 4.0);
            if (i > 0) cs_entry(A, idx, idx - size, -1.0);
            if (i < size - 1) cs_entry(A, idx, idx + size, -1.0);
            if (j > 0) cs_entry(A, idx, idx - 1, -1.0);
            if (j < size - 1) cs_entry(A, idx, idx + 1, -1.0);

            b[idx] = chemical[i][j];
        }
    }

    // Converti matrice in formato CSC
    cs* Acsc = cs_compress(A);
    cs_spfree(A);

    // Risolvi il sistema lineare in parallelo
    sparse_linear_system_cuda(Acsc, b, n);

    // Allocazione memoria GPU per i campi elettrici
    double *d_E_field_x, *d_E_field_y, *d_b;
    cudaMalloc((void**)&d_E_field_x, n * sizeof(double));
    cudaMalloc((void**)&d_E_field_y, n * sizeof(double));
    cudaMalloc((void**)&d_b, n * sizeof(double));

    // Copia del vettore b dalla CPU alla GPU
    cudaMemcpy(d_b, b, n * sizeof(double), cudaMemcpyHostToDevice);

    // Configurazione della griglia e dei blocchi per il kernel
    dim3 blockDim(16, 16);  // Blocchi di 16x16 thread
    dim3 gridDim((size + blockDim.x - 1) / blockDim.x, (size + blockDim.y - 1) / blockDim.y);

    // Lancio del kernel per calcolare i campi elettrici
    compute_fields<<<gridDim, blockDim>>>(d_b, d_E_field_x, d_E_field_y, size);

    // Sincronizzazione per garantire che il kernel sia completato
    cudaDeviceSynchronize();

    // Copia dei risultati dalla GPU alla CPU
    double* h_E_field_x = (double*)malloc(n * sizeof(double));
    double* h_E_field_y = (double*)malloc(n * sizeof(double));

    cudaMemcpy(h_E_field_x, d_E_field_x, n * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_E_field_y, d_E_field_y, n * sizeof(double), cudaMemcpyDeviceToHost);

    // Copia dei campi nei vettori 2D di output
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            int idx = i * size + j;
            E_field_x[i][j] = h_E_field_x[idx];
            E_field_y[i][j] = h_E_field_y[idx];
        }
    }

    // Cleanup
    cudaFree(d_E_field_x);
    cudaFree(d_E_field_y);
    cudaFree(d_b);
    free(h_E_field_x);
    free(h_E_field_y);
    free(b);
    cs_spfree(Acsc);
}



// Funzione per costruire il sistema lineare A * x = b con CUDA per matrici sparse
/*
__global__ void build_sparse_linear_system_kernel(int* row_indices, int* col_indices, double* values, double* b, double* chemical, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= size * size) return;

    int i = idx / size;
    int j = idx % size;

    // Valori diagonali
    row_indices[idx] = idx;
    col_indices[idx] = idx;
    values[idx] = 4.0;

    // Valori delle connessioni adiacenti (per i vicini)
    if (i > 0) {
        row_indices[idx + size] = idx;
        col_indices[idx + size] = idx - size;
        values[idx + size] = -1.0;
    }
    if (i < size - 1) {
        row_indices[idx + 2 * size] = idx;
        col_indices[idx + 2 * size] = idx + size;
        values[idx + 2 * size] = -1.0;
    }
    if (j > 0) {
        row_indices[idx + 3 * size] = idx;
        col_indices[idx + 3 * size] = idx - 1;
        values[idx + 3 * size] = -1.0;
    }
    if (j < size - 1) {
        row_indices[idx + 4 * size] = idx;
        col_indices[idx + 4 * size] = idx + 1;
        values[idx + 4 * size] = -1.0;
    }

    // Costruisci il vettore b
    b[idx] = chemical[idx];
}


*/
/*
   Questa funzione si occupa di costruire un sistema lineare sparso A*x=b utilizzando SuiteSparse 
   e di risolverlo, delegando parte della soluzione alla funzione sparse_linear_system().
   A è una matrice sparsa costruita in base alle connessioni della griglia e b è il campo chimico. 
   Una volta risolto il sistema, calcola i campi elettrici E_field_x e E_field_y 
   utilizzando le differenze finite sui valori di b.
*/

/*

void solve_sparse_linear_system_cuda(double** chemical, double** E_field_x, double** E_field_y, int size) {
    int n = size * size;
    int nnz = 5 * n; // Numero di elementi non nulli, stimato considerando una matrice con connessioni vicine (compresi i bordi)

    // Alloca memoria sulla GPU per la matrice sparsa e il vettore b
    int* d_row_indices; 
    int* d_col_indices;
    double* d_values;
    double* d_b;
    double* d_chemical;

    cudaMalloc((void**)&d_row_indices, nnz * sizeof(int));
    cudaMalloc((void**)&d_col_indices, nnz * sizeof(int));
    cudaMalloc((void**)&d_values, nnz * sizeof(double));
    cudaMalloc((void**)&d_b, n * sizeof(double));
    cudaMalloc((void**)&d_chemical, n * sizeof(double));

    // Copia chemical sulla GPU
    cudaMemcpy(d_chemical, chemical[0], n * sizeof(double), cudaMemcpyHostToDevice);

    // Configura il kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;

    // Lancia il kernel per costruire la matrice sparsa e il vettore b
    build_sparse_linear_system_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_row_indices, d_col_indices, d_values, d_b, d_chemical, size);
    cudaDeviceSynchronize();

    // Risolvere il sistema sparso con cuSPARSE
    cusparseHandle_t handle;
    cusparseCreate(&handle);

    // Descrittori della matrice
    cusparseMatDescr_t descrA;
    cusparseCreateMatDescr(&descrA);
    cusparseSetMatType(descrA, CUSPARSE_MATRIX_TYPE_GENERAL);
    cusparseSetMatIndexBase(descrA, CUSPARSE_INDEX_BASE_ZERO);

    // Passo 1: Analisi della matrice (fase di pre-elaborazione)
    csrsv2Info_t info = NULL;
    cusparseCreateCsrsv2Info(&info);

    int bufferSize = 0;
    void* pBuffer = NULL;

    cusparseDcsrsv2_bufferSize(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, n, nnz, descrA, d_values, d_row_indices, d_col_indices, info, &bufferSize);
    cudaMalloc((void**)&pBuffer, bufferSize);

    // Passo 2: Analisi della matrice (fase di setup)
    cusparseDcsrsv2_analysis(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, n, nnz, descrA, d_values, d_row_indices, d_col_indices, info, CUSPARSE_SOLVE_POLICY_NO_LEVEL, pBuffer);

    // Alloca memoria per il vettore soluzione
    double* d_x;
    cudaMalloc((void**)&d_x, n * sizeof(double));

    // Passo 3: Risoluzione del sistema A * x = b
    const double alpha = 1.0;
    cusparseDcsrsv2_solve(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, n, nnz, &alpha, descrA, d_values, d_row_indices, d_col_indices, info, d_b, d_x, CUSPARSE_SOLVE_POLICY_NO_LEVEL, pBuffer);

    // Copia il risultato dalla GPU alla CPU
    double* h_x = (double*)malloc(n * sizeof(double));
    cudaMemcpy(h_x, d_x, n * sizeof(double), cudaMemcpyDeviceToHost);

    // Calcola i campi E_field_x e E_field_y
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

    // Libera la memoria
    cudaFree(d_row_indices);
    cudaFree(d_col_indices);
    cudaFree(d_values);
    cudaFree(d_b);
    cudaFree(d_chemical);
    cudaFree(d_x);
    cudaFree(pBuffer);
    cusparseDestroyMatDescr(descrA);
    cusparseDestroyCsrsv2Info(info);
    cusparseDestroy(handle);
    free(h_x);
}

// Funzione per calcolare il cambiamento di energia di adesione con CUDA
__global__ void calculate_delta_adhesion_kernel(int* d_lattice, double* d_delta_H, int nx, int ny, int size) {
    // Calcola gli indici x e y basati sugli indici di thread e blocco
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= size || y >= size) return;

    int current_cell = d_lattice[y * size + x];
    int neighbor_cell = d_lattice[ny * size + nx];  // Ricordati che `nx` e `ny` sono ora passati come argomenti
    double delta_H = 0.0;

    // Parametri per gestire le condizioni periodiche
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

            int current_neighbor_cell = d_lattice[yy * size + xx];

            // Se il vicino ha un ID diverso dalla cella corrente, diminuisci l'energia
            if (current_neighbor_cell != current_cell) {
                delta_H -= 1.0;
            }

            // Aumenta l'energia se il vicino del candidato è diverso
            if (current_neighbor_cell != neighbor_cell) {
                delta_H += 1.0;
            }
        }
    }

    // Salva il risultato nel vettore di output
    d_delta_H[y * size + x] = delta_H;
}

// Funzione host per lanciare il kernel per calcolare il cambiamento di energia di adesione
void calculate_delta_adhesion_cuda(int* h_lattice, double* h_delta_H, int nx, int ny, int size) {
    // Allocazione della memoria sul dispositivo
    int* d_lattice;
    double* d_delta_H;

    cudaMalloc((void**)&d_lattice, size * size * sizeof(int));
    cudaMalloc((void**)&d_delta_H, size * size * sizeof(double));

    // Copia dei dati dalla CPU alla GPU
    cudaMemcpy(d_lattice, h_lattice, size * size * sizeof(int), cudaMemcpyHostToDevice);

    // Configurazione del kernel: definizione della dimensione dei blocchi e della griglia
    dim3 blockSize(16, 16);  // Dimensione del blocco
    dim3 gridSize((size + blockSize.x - 1) / blockSize.x, (size + blockSize.y - 1) / blockSize.y);  // Dimensione della griglia

    // Lancio del kernel per calcolare il cambiamento di energia di adesione
    calculate_delta_adhesion_kernel<<<gridSize, blockSize>>>(d_lattice, d_delta_H, nx, ny, size);

    // Sincronizza i thread
    cudaDeviceSynchronize();

    // Copia del risultato dalla GPU alla CPU
    cudaMemcpy(h_delta_H, d_delta_H, size * size * sizeof(double), cudaMemcpyDeviceToHost);

    // Liberazione della memoria sulla GPU
    cudaFree(d_lattice);
    cudaFree(d_delta_H);
}

// Funzione per calcolare il cambiamento di energia di volume con CUDA
__global__ void calculate_delta_volume_kernel(double* d_delta_H, int* d_current_volume, int* d_neighbor_volume, int target_volume, int num_elements) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= num_elements) return;

    int current_volume = d_current_volume[idx];
    int neighbor_volume = d_neighbor_volume[idx];

    // Calcolo delta_H per il volume utilizzando moltiplicazioni al posto di pow() per evitare l'errore calling a __host__ function
    double delta_H = LAMBDA_V * (
        (neighbor_volume + 1 - target_volume) * (neighbor_volume + 1 - target_volume) +
        (current_volume - 1 - target_volume) * (current_volume - 1 - target_volume) -
        (neighbor_volume - target_volume) * (neighbor_volume - target_volume) -
        (current_volume - target_volume) * (current_volume - target_volume)
    );


    // Salva il risultato
    d_delta_H[idx] = delta_H;
}

// Funzione host per lanciare il kernel per calcolare il cambiamento di energia di volume
void calculate_delta_volume_cuda(int* h_current_volume, int* h_neighbor_volume, double* h_delta_H, int target_volume, int num_elements) {
    // Allocazione della memoria sul dispositivo
    int* d_current_volume;
    int* d_neighbor_volume;
    double* d_delta_H;

    cudaMalloc((void**)&d_current_volume, num_elements * sizeof(int));
    cudaMalloc((void**)&d_neighbor_volume, num_elements * sizeof(int));
    cudaMalloc((void**)&d_delta_H, num_elements * sizeof(double));

    // Copia dei dati dalla CPU alla GPU
    cudaMemcpy(d_current_volume, h_current_volume, num_elements * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_neighbor_volume, h_neighbor_volume, num_elements * sizeof(int), cudaMemcpyHostToDevice);

    // Configurazione del kernel: definizione della dimensione dei blocchi e della griglia
    int blockSize = 256;  // Dimensione del blocco
    int gridSize = (num_elements + blockSize - 1) / blockSize;  // Dimensione della griglia

    // Lancio del kernel per calcolare il cambiamento di energia di volume
    calculate_delta_volume_kernel<<<gridSize, blockSize>>>(d_delta_H, d_current_volume, d_neighbor_volume, target_volume, num_elements);

    // Sincronizza i thread
    cudaDeviceSynchronize();

    // Copia del risultato dalla GPU alla CPU
    cudaMemcpy(h_delta_H, d_delta_H, num_elements * sizeof(double), cudaMemcpyDeviceToHost);

    // Liberazione della memoria sulla GPU
    cudaFree(d_current_volume);
    cudaFree(d_neighbor_volume);
    cudaFree(d_delta_H);
}

// Funzione per calcolare il cambiamento di energia di superficie con CUDA
__global__ void calculate_delta_surface_kernel(int* d_lattice, double* d_delta_S, int size, int nx, int ny) {
    // Indici del thread corrente
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= size || y >= size) return;

    // Identificare la cella corrente e quella del vicino
    int current_cell = d_lattice[x * size + y];
    int neighbor_cell = d_lattice[nx * size + ny];

    double delta_S_current = 0.0;
    double delta_S_neighbor = 0.0;

    // Calcolare i contributi dell'energia di superficie
    for (int i = -1; i <= 1; i++) {
        for (int j = -1; j <= 1; j++) {
            if (i == 0 && j == 0) continue;

            int xx = (x + i + size) % size;
            int yy = (y + j + size) % size;

            if (d_lattice[xx * size + yy] != current_cell) {
                delta_S_current--;
            }
            if (d_lattice[xx * size + yy] != neighbor_cell) {
                delta_S_neighbor++;
            }
        }
    }

    // Salvataggio del risultato nel vettore delta_S
    d_delta_S[x * size + y] = LAMBDA_S * (delta_S_current + delta_S_neighbor);
}

// Funzione host per lanciare il kernel per calcolare il cambiamento di energia di superficie
void calculate_delta_surface_cuda(int* h_lattice, double* h_delta_S, int size, int nx, int ny) {
    // Allocazione della memoria sul dispositivo
    int* d_lattice;
    double* d_delta_S;

    cudaMalloc((void**)&d_lattice, size * size * sizeof(int));
    cudaMalloc((void**)&d_delta_S, size * size * sizeof(double));

    // Copia dei dati dalla CPU alla GPU
    cudaMemcpy(d_lattice, h_lattice, size * size * sizeof(int), cudaMemcpyHostToDevice);

    // Configurazione del kernel: definizione della dimensione dei blocchi e della griglia
    dim3 blockSize(16, 16);  // Dimensione del blocco
    dim3 gridSize((size + blockSize.x - 1) / blockSize.x, (size + blockSize.y - 1) / blockSize.y);  // Dimensione della griglia

    // Lancio del kernel per calcolare il cambiamento di energia di superficie
    calculate_delta_surface_kernel<<<gridSize, blockSize>>>(d_lattice, d_delta_S, size, nx, ny);

    // Sincronizza i thread
    cudaDeviceSynchronize();

    // Copia del risultato dalla GPU alla CPU
    cudaMemcpy(h_delta_S, d_delta_S, size * size * sizeof(double), cudaMemcpyDeviceToHost);

    // Liberazione della memoria sulla GPU
    cudaFree(d_lattice);
    cudaFree(d_delta_S);
}

// Funzione per calcolare il cambiamento di energia di chemiotassi con CUDA
__global__ void calculate_delta_chemotaxis_kernel(double* d_chemical, double* d_delta_H, int size) {
    // Indici del thread corrente
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= size || y >= size) return;

    // Calcolo delle coordinate del vicino
    // Qui assumiamo che `(nx, ny)` sia un vicino in una certa direzione
    // Per esempio, consideriamo il vicino a destra (x, y+1)
    int nx = (x + 1) % size; // Per un vicino a destra
    int ny = (y + 1) % size; // Per un vicino in alto

    // Calcolo del cambiamento di energia di chemiotassi
    double delta_H = -LAMBDA_C * (d_chemical[nx * size + ny] - d_chemical[x * size + y]);

    // Salva il risultato nella matrice di output
    d_delta_H[x * size + y] = delta_H;
}

// Funzione host per lanciare il kernel per calcolare il cambiamento di energia di chemiotassi
void calculate_delta_chemotaxis_cuda(double* h_chemical, double* h_delta_H, int size) {
    // Allocazione della memoria sul dispositivo
    double* d_chemical;
    double* d_delta_H;

    cudaMalloc((void**)&d_chemical, size * size * sizeof(double));
    cudaMalloc((void**)&d_delta_H, size * size * sizeof(double));

    // Copia dei dati dalla CPU alla GPU
    cudaMemcpy(d_chemical, h_chemical, size * size * sizeof(double), cudaMemcpyHostToDevice);

    // Configurazione del kernel: definizione della dimensione dei blocchi e della griglia
    dim3 blockSize(16, 16);  // Dimensione del blocco
    dim3 gridSize((size + blockSize.x - 1) / blockSize.x, (size + blockSize.y - 1) / blockSize.y);  // Dimensione della griglia

    // Lancio del kernel per calcolare il cambiamento di energia di chemiotassi
    calculate_delta_chemotaxis_kernel<<<gridSize, blockSize>>>(d_chemical, d_delta_H, size);

    // Sincronizza i thread
    cudaDeviceSynchronize();

    // Copia del risultato dalla GPU alla CPU
    cudaMemcpy(h_delta_H, d_delta_H, size * size * sizeof(double), cudaMemcpyDeviceToHost);

    // Liberazione della memoria sulla GPU
    cudaFree(d_chemical);
    cudaFree(d_delta_H);
}

// Funzione per calcolare il cambiamento di energia del campo elettrico con CUDA
__global__ void calculate_delta_electric_kernel(int* d_x, int* d_y, int* d_nx, int* d_ny, double* d_E_field_x, double* d_E_field_y, double* d_delta_H, int size) {
    // Ottieni l'indice globale del thread
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Assicurati che l'indice non superi la dimensione del sistema
    if (idx >= size) return;

    // Ottieni le coordinate x, y, nx, ny dal dispositivo
    int x = d_x[idx];
    int y = d_y[idx];
    int nx = d_nx[idx];
    int ny = d_ny[idx];

    // Calcola le differenze nelle coordinate
    int delta_x = nx - x;
    int delta_y = ny - y;

    // Calcola il cambiamento di energia del campo elettrico
    d_delta_H[idx] = -LAMBDA_E * (d_E_field_x[x * size + y] * delta_x + d_E_field_y[x * size + y] * delta_y);
}

// Funzione host per lanciare il kernel
void calculate_delta_electric_cuda(int* h_x, int* h_y, int* h_nx, int* h_ny, double** E_field_x, double** E_field_y, double* h_delta_H, int num_elements, int size) {
    // Allocazione memoria sul dispositivo
    int* d_x; int* d_y; int* d_nx; int* d_ny;
    double* d_E_field_x; double* d_E_field_y; double* d_delta_H;

    cudaMalloc((void**)&d_x, num_elements * sizeof(int));
    cudaMalloc((void**)&d_y, num_elements * sizeof(int));
    cudaMalloc((void**)&d_nx, num_elements * sizeof(int));
    cudaMalloc((void**)&d_ny, num_elements * sizeof(int));
    cudaMalloc((void**)&d_E_field_x, size * size * sizeof(double));
    cudaMalloc((void**)&d_E_field_y, size * size * sizeof(double));
    cudaMalloc((void**)&d_delta_H, num_elements * sizeof(double));

    // Copia dei dati dalla CPU alla GPU
    cudaMemcpy(d_x, h_x, num_elements * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, h_y, num_elements * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_nx, h_nx, num_elements * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_ny, h_ny, num_elements * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_E_field_x, E_field_x[0], size * size * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_E_field_y, E_field_y[0], size * size * sizeof(double), cudaMemcpyHostToDevice);

    // Definizione della configurazione del kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (num_elements + threadsPerBlock - 1) / threadsPerBlock;

    // Lancio del kernel
    calculate_delta_electric_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_x, d_y, d_nx, d_ny, d_E_field_x, d_E_field_y, d_delta_H, size);

    // Sincronizza i thread
    cudaDeviceSynchronize();

    // Copia del risultato dalla GPU alla CPU
    cudaMemcpy(h_delta_H, d_delta_H, num_elements * sizeof(double), cudaMemcpyDeviceToHost);

    // Liberazione della memoria sul dispositivo
    cudaFree(d_x);
    cudaFree(d_y);
    cudaFree(d_nx);
    cudaFree(d_ny);
    cudaFree(d_E_field_x);
    cudaFree(d_E_field_y);
    cudaFree(d_delta_H);
}

// Funzione per calcolare l'energia totale con CUDA
__global__ void calculate_delta_total_energy_kernel(int* d_lattice, double* d_chemical, double* d_E_field_x, double* d_E_field_y, int* d_x, int* d_y, int* d_nx, int* d_ny, int* d_current_volume, int* d_neighbor_volume, double* d_delta_H_total, double* d_delta_H_adhesion, double* d_delta_H_volume, double* d_delta_H_surface, double* d_delta_H_chemotaxis, double* d_delta_H_electric, int target_volume, int size, int num_elements) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= num_elements) return;

    int x = d_x[idx];
    int y = d_y[idx];
    int nx = d_nx[idx];
    int ny = d_ny[idx];

    if (d_lattice[x * size + y] == d_lattice[nx * size + ny]) {
        d_delta_H_total[idx] = 0.0;  // Nessun cambiamento se le celle sono uguali
        return;
    }

    // Calcola l'energia di adesione
    calculate_delta_adhesion_kernel<<<1, 1>>>(d_lattice, &d_delta_H_adhesion[idx], size, nx, ny);
    cudaDeviceSynchronize();

    // Calcola l'energia di volume se necessario
    if (!(d_lattice[x * size + y] == 0 || d_lattice[nx * size + ny] == 0)) {
        calculate_delta_volume_kernel<<<1, 1>>>(&d_delta_H_volume[idx], &d_current_volume[idx], &d_neighbor_volume[idx], target_volume, 1);
        cudaDeviceSynchronize();

        calculate_delta_surface_kernel<<<1, 1>>>(d_lattice, &d_delta_H_surface[idx], size, nx, ny);
        cudaDeviceSynchronize();
    } else {
        d_delta_H_volume[idx] = 0.0;
        d_delta_H_surface[idx] = 0.0;
    }

    // Calcola l'energia di chemiotassi
    calculate_delta_chemotaxis_kernel<<<1, 1>>>(d_chemical, &d_delta_H_chemotaxis[idx], size);
    cudaDeviceSynchronize();

    // Calcola l'energia elettrica
    calculate_delta_electric_kernel<<<1, 1>>>(d_x, d_y, d_nx, d_ny, d_E_field_x, d_E_field_y, &d_delta_H_electric[idx], size);
    cudaDeviceSynchronize();

    // Somma dei contributi energetici
    d_delta_H_total[idx] = d_delta_H_adhesion[idx] + d_delta_H_volume[idx] + d_delta_H_surface[idx] + d_delta_H_chemotaxis[idx] + d_delta_H_electric[idx];
}

// Funzione host per lanciare il kernel
void calculate_delta_total_energy_cuda(int* h_lattice, double** chemical, double** E_field_x, double** E_field_y, int* h_x, int* h_y, int* h_nx, int* h_ny, int* h_current_volume, int* h_neighbor_volume, int target_volume, double* h_delta_H_total, int num_elements, int size) {
    // Allocazione della memoria sul dispositivo
    int* d_lattice; double* d_chemical; double* d_E_field_x; double* d_E_field_y;
    int* d_x; int* d_y; int* d_nx; int* d_ny;
    int* d_current_volume; int* d_neighbor_volume;
    double* d_delta_H_total;
    double* d_delta_H_adhesion; double* d_delta_H_volume; double* d_delta_H_surface; double* d_delta_H_chemotaxis; double* d_delta_H_electric;

    cudaMalloc((void**)&d_lattice, size * size * sizeof(int));
    cudaMalloc((void**)&d_chemical, size * size * sizeof(double));
    cudaMalloc((void**)&d_E_field_x, size * size * sizeof(double));
    cudaMalloc((void**)&d_E_field_y, size * size * sizeof(double));
    cudaMalloc((void**)&d_x, num_elements * sizeof(int));
    cudaMalloc((void**)&d_y, num_elements * sizeof(int));
    cudaMalloc((void**)&d_nx, num_elements * sizeof(int));
    cudaMalloc((void**)&d_ny, num_elements * sizeof(int));
    cudaMalloc((void**)&d_current_volume, num_elements * sizeof(int));
    cudaMalloc((void**)&d_neighbor_volume, num_elements * sizeof(int));
    cudaMalloc((void**)&d_delta_H_total, num_elements * sizeof(double));
    cudaMalloc((void**)&d_delta_H_adhesion, num_elements * sizeof(double));
    cudaMalloc((void**)&d_delta_H_volume, num_elements * sizeof(double));
    cudaMalloc((void**)&d_delta_H_surface, num_elements * sizeof(double));
    cudaMalloc((void**)&d_delta_H_chemotaxis, num_elements * sizeof(double));
    cudaMalloc((void**)&d_delta_H_electric, num_elements * sizeof(double));

    // Copia dei dati dalla CPU alla GPU
    cudaMemcpy(d_lattice, h_lattice, size * size * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_chemical, chemical[0], size * size * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_E_field_x, E_field_x[0], size * size * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_E_field_y, E_field_y[0], size * size * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_x, h_x, num_elements * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, h_y, num_elements * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_nx, h_nx, num_elements * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_ny, h_ny, num_elements * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_current_volume, h_current_volume, num_elements * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_neighbor_volume, h_neighbor_volume, num_elements * sizeof(int), cudaMemcpyHostToDevice);

    // Configurazione del kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (num_elements + threadsPerBlock - 1) / threadsPerBlock;

    // Lancio del kernel per calcolare il delta di energia totale
    calculate_delta_total_energy_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_lattice, d_chemical, d_E_field_x, d_E_field_y, d_x, d_y, d_nx, d_ny, d_current_volume, d_neighbor_volume, d_delta_H_total, d_delta_H_adhesion, d_delta_H_volume, d_delta_H_surface, d_delta_H_chemotaxis, d_delta_H_electric, target_volume, size, num_elements);

    // Sincronizza i thread
    cudaDeviceSynchronize();

    // Copia del risultato dalla GPU alla CPU
    cudaMemcpy(h_delta_H_total, d_delta_H_total, num_elements * sizeof(double), cudaMemcpyHostToDevice);

    // Liberazione della memoria sulla GPU
    cudaFree(d_lattice);
    cudaFree(d_chemical);
    cudaFree(d_E_field_x);
    cudaFree(d_E_field_y);
    cudaFree(d_x);
    cudaFree(d_y);
    cudaFree(d_nx);
    cudaFree(d_ny);
    cudaFree(d_current_volume);
    cudaFree(d_neighbor_volume);
    cudaFree(d_delta_H_total);
    cudaFree(d_delta_H_adhesion);
    cudaFree(d_delta_H_volume);
    cudaFree(d_delta_H_surface);
    cudaFree(d_delta_H_chemotaxis);
    cudaFree(d_delta_H_electric);
}



// Funzione per inizializzare il campo elettrico con CUDA
__global__ void initialize_electric_field_kernel(double* d_E_field_x, double* d_E_field_y, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;

    if (idx < size && idy < size) {
        d_E_field_x[idx * size + idy] = 0.0;
        d_E_field_y[idx * size + idy] = 0.0;
    }
}

// Funzione host per lanciare il kernel
void initialize_electric_field_cuda(double** E_field_x, double** E_field_y, int size) {
    double* d_E_field_x;
    double* d_E_field_y;

    cudaMalloc((void**)&d_E_field_x, size * size * sizeof(double));
    cudaMalloc((void**)&d_E_field_y, size * size * sizeof(double));

    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((size + threadsPerBlock.x - 1) / threadsPerBlock.x, (size + threadsPerBlock.y - 1) / threadsPerBlock.y);

    initialize_electric_field_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_E_field_x, d_E_field_y, size);
    cudaDeviceSynchronize();

    cudaMemcpy(E_field_x[0], d_E_field_x, size * size * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(E_field_y[0], d_E_field_y, size * size * sizeof(double), cudaMemcpyDeviceToHost);

    cudaFree(d_E_field_x);
    cudaFree(d_E_field_y);
}

// Funzione per trovare il valore massimo nella matrice chimica con CUDA
__global__ void find_max_chemical_kernel(double* d_chemical, double* d_max, int size) {
    extern __shared__ double sdata[];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Inizializza la memoria condivisa
    if (idx < size * size) {
        sdata[tid] = d_chemical[idx];
    } else {
        sdata[tid] = -DBL_MAX;  // Assicura che non influenzi il risultato del massimo
    }
    __syncthreads();

    // Riduzione parallela per trovare il massimo
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s && idx + s < size * size) {
            sdata[tid] = max(sdata[tid], sdata[tid + s]);
        }
        __syncthreads();
    }

    // Salva il risultato nel vettore di output
    if (tid == 0) {
        d_max[blockIdx.x] = sdata[0];
    }
}

// Funzione host per lanciare il kernel
double find_max_chemical_cuda(double** chemical, int size) {
    double* d_chemical;
    double* d_max;
    double* h_max;
    int blockSize = 256;
    int gridSize = (size * size + blockSize - 1) / blockSize;

    // Alloca la memoria sulla GPU
    cudaMalloc((void**)&d_chemical, size * size * sizeof(double));
    cudaMalloc((void**)&d_max, gridSize * sizeof(double));
    h_max = (double*)malloc(gridSize * sizeof(double));

    // Copia i dati dalla CPU alla GPU
    cudaMemcpy(d_chemical, chemical[0], size * size * sizeof(double), cudaMemcpyHostToDevice);

    // Configura la memoria condivisa e lancia il kernel
    int sharedMemorySize = blockSize * sizeof(double);
    find_max_chemical_kernel<<<gridSize, blockSize, sharedMemorySize>>>(d_chemical, d_max, size);

    // Sincronizza i thread
    cudaDeviceSynchronize();

    // Copia i risultati dalla GPU alla CPU
    cudaMemcpy(h_max, d_max, gridSize * sizeof(double), cudaMemcpyDeviceToHost);

    // Riduzione finale sulla CPU per trovare il massimo
    double max_value = h_max[0];
    for (int i = 1; i < gridSize; i++) {
        if (h_max[i] > max_value) {
            max_value = h_max[i];
        }
    }

    // Libera la memoria sulla GPU e sulla CPU
    cudaFree(d_chemical);
    cudaFree(d_max);
    free(h_max);

    return max_value;
}

// Funzione per normalizzare il campo chimico con CUDA
__global__ void normalize_chemical_field_kernel(double* d_chemical, double max_value, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size * size && max_value > 0) {
        d_chemical[idx] /= max_value;
    }
}

// Funzione host per lanciare il kernel
void normalize_chemical_field_cuda(double** chemical, int size) {
    double* d_chemical;
    double max_value;

    // Trova il valore massimo usando la funzione precedentemente definita
    max_value = find_max_chemical_cuda(chemical, size);

    // Alloca la memoria sulla GPU
    cudaMalloc((void**)&d_chemical, size * size * sizeof(double));

    // Copia i dati dalla CPU alla GPU
    cudaMemcpy(d_chemical, chemical[0], size * size * sizeof(double), cudaMemcpyHostToDevice);

    // Configura il kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (size * size + threadsPerBlock - 1) / threadsPerBlock;

    // Lancia il kernel per normalizzare il campo chimico
    normalize_chemical_field_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_chemical, max_value, size);

    // Sincronizza i thread
    cudaDeviceSynchronize();

    // Copia i dati dalla GPU alla CPU
    cudaMemcpy(chemical[0], d_chemical, size * size * sizeof(double), cudaMemcpyDeviceToHost);

    // Libera la memoria sulla GPU
    cudaFree(d_chemical);
}


// Funzione per aggiornare il campo chimico
// Fa uso di puntatori a puntatori per evitare di copiare la matrice, quindi differisce dalla versione Matlab
// Per comodità si fa uso di due puntatori a matrice, uno per la matrice corrente e uno per la matrice successiva

__global__ void update_chemical_kernel_shared(double* d_chemical, int* d_lattice, double* d_new_chemical, int size) {
    // Calcola gli indici del thread in base al blocco e all'ID del thread
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    // Definizione della memoria condivisa per il blocco, con padding per evitare accessi fuori dai limiti
    extern __shared__ double shared_chemical[];

    // Calcola l'indice del thread nel blocco condiviso
    int local_x = threadIdx.x + 1; // 1 per padding per la gestione dei bordi
    int local_y = threadIdx.y + 1; // 1 per padding per la gestione dei bordi

    // Copia la cella corrente nella memoria condivisa
    if (x < size && y < size) {
        shared_chemical[local_y * (blockDim.x + 2) + local_x] = d_chemical[y * size + x];
        
        // Copia anche le celle del bordo (padding) per il calcolo del laplaciano
        if (threadIdx.x == 0 && x > 0) {
            shared_chemical[local_y * (blockDim.x + 2)] = d_chemical[y * size + (x - 1)];
        }
        if (threadIdx.x == blockDim.x - 1 && x < size - 1) {
            shared_chemical[local_y * (blockDim.x + 2) + (local_x + 1)] = d_chemical[y * size + (x + 1)];
        }
        if (threadIdx.y == 0 && y > 0) {
            shared_chemical[local_y * (blockDim.x + 2) + local_x - 1] = d_chemical[(y - 1) * size + x];
        }
        if (threadIdx.y == blockDim.y - 1 && y < size - 1) {
            shared_chemical[(local_y + 1) * (blockDim.x + 2) + local_x] = d_chemical[(y + 1) * size + x];
        }
    }

    // Sincronizza tutti i thread nel blocco per assicurarsi che la memoria condivisa sia completamente aggiornata
    __syncthreads();

    if (x >= size || y >= size) return;

    // Calcolo del laplaciano usando i valori in memoria condivisa
    double laplacian = -4 * shared_chemical[local_y * (blockDim.x + 2) + local_x];
    laplacian += shared_chemical[local_y * (blockDim.x + 2) + (local_x - 1)];
    laplacian += shared_chemical[local_y * (blockDim.x + 2) + (local_x + 1)];
    laplacian += shared_chemical[(local_y - 1) * (blockDim.x + 2) + local_x];
    laplacian += shared_chemical[(local_y + 1) * (blockDim.x + 2) + local_x];

    // Calcolo del nuovo valore chimico
    d_new_chemical[y * size + x] = d_chemical[y * size + x] + DIFF_DT * (
                                   D * laplacian -
                                   DECAY_RATE * d_chemical[y * size + x] +
                                   SECRETION_RATE * (d_lattice[y * size + x] > 0 ? 1 : 0));
}

// Funzione host per lanciare il kernel con memoria condivisa per l'aggiornamento chimico
void update_chemical_cuda_shared(double* h_chemical, int* h_lattice, double* h_new_chemical, int size) {
    // Allocazione della memoria sul dispositivo
    double* d_chemical;
    int* d_lattice;
    double* d_new_chemical;

    cudaMalloc((void**)&d_chemical, size * size * sizeof(double));
    cudaMalloc((void**)&d_lattice, size * size * sizeof(int));
    cudaMalloc((void**)&d_new_chemical, size * size * sizeof(double));

    // Copia dei dati dalla CPU alla GPU
    cudaMemcpy(d_chemical, h_chemical, size * size * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_lattice, h_lattice, size * size * sizeof(int), cudaMemcpyHostToDevice);

    // Configurazione del kernel: definizione della dimensione dei blocchi e della griglia
    dim3 blockSize(16, 16);  // Dimensione del blocco
    dim3 gridSize((size + blockSize.x - 1) / blockSize.x, (size + blockSize.y - 1) / blockSize.y);  // Dimensione della griglia

    // Calcolo della dimensione della memoria condivisa necessaria per ogni blocco
    int sharedMemSize = (blockSize.x + 2) * (blockSize.y + 2) * sizeof(double);

    // Lancio del kernel per l'aggiornamento del campo chimico
    update_chemical_kernel_shared<<<gridSize, blockSize, sharedMemSize>>>(d_chemical, d_lattice, d_new_chemical, size);

    // Sincronizza i thread
    cudaDeviceSynchronize();

    // Copia del risultato dalla GPU alla CPU
    cudaMemcpy(h_new_chemical, d_new_chemical, size * size * sizeof(double), cudaMemcpyDeviceToHost);

    // Liberazione della memoria sulla GPU
    cudaFree(d_chemical);
    cudaFree(d_lattice);
    cudaFree(d_new_chemical);
}




*/



/* DA PARALLELIZZARE
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

void free_matrix(double** matrix, int size) {
    for (int i = 0; i < size; i++) {
        free(matrix[i]);
    }
    free(matrix);
}

void free_int_matrix(int** matrix, int size) {
    for (int i = 0; i < size; i++) {
        free(matrix[i]);
    }
    free(matrix);
}
*/ 