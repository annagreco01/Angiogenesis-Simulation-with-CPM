/*
gcc cpm.c calc_time.c -o cpm.out -I/home/greco/miniconda3/envs/sparsematrix/include/suitesparse -L/home/greco/miniconda3/envs/sparsematrix/lib -lcxsparse -lsuitesparseconfig -llapacke -llapack -lblas -Wl,-rpath=/home/greco/miniconda3/envs/sparsematrix/lib -std=c11 -lm
*/

#include "cpm.h"

int cell_volumes[NUM_CELLS + 1] = {0};

int main(int argc, char* argv[]) {
    printf("**************************************************************************************************************\n\n");
    printf(" Per utilizzare le matrici dense con LAPACKE_dgesv rilanciare il programma con il parametro [dense] \n");
    printf(" Altrimenti verrà usata la libreria SparseSuite\n");
    printf("**************************************************************************************************************\n");
    printf(" Esecuzione con %d celle e %d MCS\n", NUM_CELLS, NUM_MCS);
    printf(" Su una griglia %d x %d\n", LATTICE_SIZE, LATTICE_SIZE);
    printf(" Con target volume %d e temperatura %.2f\n", TARGET_VOLUME, T);
    int flagDense = 0;
    // Inizializzazione del generatore di numeri casuali
    if(argc > 1 && strcmp(argv[1], "dense") == 0) {
        flagDense = 1;
        printf(" Elaborazione con le matrici dense\n");

    } else {
            printf(" Elaborazione con le matrici sparse\n");
    }
    printf("**************************************************************************************************************\n\n");

    srand(time(NULL));
    // inizia a prendere il calc_time
    calc_timeStart();
    // Parametri del campo elettrico
    double** E_field_x = allocate_matrix(LATTICE_SIZE);
    double** E_field_y = allocate_matrix(LATTICE_SIZE);
    // Dichiarazione delle matrici per il reticolo e il campo chimico
    double** chemical = allocate_matrix(LATTICE_SIZE);
    int** lattice = allocate_int_matrix(LATTICE_SIZE);
    int history_size = floor(NUM_MCS/10) + 1;
    int frame_count = 0;
    int*** lattice_history = (int***) malloc(history_size * sizeof(int**));
    double*** chemical_history = (double***) malloc(history_size * sizeof(double**));

    // Inizializzazione delle celle nel reticolo
    initialize_cells(lattice);

    // Ciclo principale di simulazione Monte Carlo
    for (int mcs = 0; mcs < NUM_MCS; mcs++) {
        printf("MCS time step: %d of %d\n", mcs + 1, NUM_MCS);
        calc_timeCheck();
        calc_timePrint();

            // Salva lo stato del reticolo per la visualizzazione
            // comincia dallo stato iniziale e continua ogni 10 passi
            if (mcs % 10 == 0) {
                lattice_history[frame_count] = allocate_int_matrix(LATTICE_SIZE);
                chemical_history[frame_count] = allocate_matrix(LATTICE_SIZE);
                for (int i = 0; i < LATTICE_SIZE; i++) {
                    for (int j = 0; j < LATTICE_SIZE; j++) {
                        lattice_history[frame_count][i][j] = lattice[i][j];
                        chemical_history[frame_count][i][j] = chemical[i][j];
                    }
                }
                frame_count++;
            }

        // Esegui un passo Monte Carlo per ogni sito del reticolo
        for (int i = 0; i < LATTICE_SIZE * LATTICE_SIZE; i++) {

            // Seleziona casualmente un sito del reticolo
            int x = rand() % LATTICE_SIZE;
            int y = rand() % LATTICE_SIZE;
            
            // Seleziona casualmente un vicino
            int nx, ny;
            get_random_neighbor(x, y, &nx, &ny);
            
            // Volume corrente della cella
            int cell_id = lattice[x][y];
            //if (cell_id != 0) {  // Se la cella non è vuota
                int current_volume = cell_id == 0 ? 0 : cell_volumes[cell_id];
                int neighbor_id = lattice[nx][ny];
                int neighbor_volume = neighbor_id == 0 ? 0 : cell_volumes[neighbor_id];
                // Calcola il cambiamento di energia per una possibile transizione
                double delta_H = calculate_delta_total_energy(lattice, chemical, E_field_x, E_field_y, x, y, nx, ny, current_volume, neighbor_volume, TARGET_VOLUME, LATTICE_SIZE);
                
                // Applica l'algoritmo di Metropolis
                double fattore = exp(-delta_H/T);
                double caso = (double)rand() / RAND_MAX;
                if (delta_H <= 0 || caso < fattore) {
                    // Transizione accettata
                    lattice[x][y] = lattice[nx][ny];
                    // Aggiorna i volumi delle celle coinvolte
                    if (cell_id != 0) {
                        cell_volumes[cell_id]--; // Diminuisci il volume della vecchia posizione
                    }
                    if (neighbor_id != 0) {
                        cell_volumes[neighbor_id]++;  // Aumenta il volume della nuova posizione
                    }
                }
            //}
        }
        
        // Aggiorna il campo elettrico dopo ogni MCS
        if (flagDense){
            solve_linear_system(chemical, E_field_x, E_field_y, LATTICE_SIZE); //Fa uso di LAPACKE con le matrici dense
        } else {
            solve_sparse_linear_system(chemical, E_field_x, E_field_y, LATTICE_SIZE); // Fa uso di SuiteSparse con le matrici sparse
        }
        
        // Aggiorna il campo chimico e lo normalizza dopo ogni MCS
        update_chemical(chemical, lattice, LATTICE_SIZE);
        normalize_chemical_field(chemical, LATTICE_SIZE);

    }
    // stato finale
    lattice_history[frame_count] = allocate_int_matrix(LATTICE_SIZE);
    chemical_history[frame_count] = allocate_matrix(LATTICE_SIZE);
    for (int i = 0; i < LATTICE_SIZE; i++) {
        for (int j = 0; j < LATTICE_SIZE; j++) {
            lattice_history[frame_count][i][j] = lattice[i][j];
            chemical_history[frame_count][i][j] = chemical[i][j];
        }
    }
    frame_count++;
    
    printf("**************************************************************************************************************\n\n");
    printf("Simulazione completata, sto per scrivere su file.\n");
    // Salva i dati su file in formato compatibile con MATLAB
    write_matrix(lattice_history, chemical_history, frame_count, LATTICE_SIZE);
    printf("Ho stampato su file.\n");
    // Liberazione della memoria
    free_matrix(E_field_x, LATTICE_SIZE);
    free_matrix(E_field_y, LATTICE_SIZE);
    free_int_matrix(lattice, LATTICE_SIZE);
    free_matrix(chemical, LATTICE_SIZE);
    for (int i = 1; i < frame_count; i++) {
        free_int_matrix(lattice_history[i], LATTICE_SIZE);
        free_matrix(chemical_history[i], LATTICE_SIZE);
    }
    free(lattice_history);
    free(chemical_history);

    printf("Simulazione completata!\n");
    calc_timeCheck();
    calc_timePrint();
    return 0;
}
