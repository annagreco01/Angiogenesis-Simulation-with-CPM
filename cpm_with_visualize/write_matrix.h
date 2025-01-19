#ifndef WRITE_MATRIX_H
#define WRITE_MATRIX_H

#include <stdio.h>
//#define LATTICE_SIZE 200
//#define NUM_MATRICI 10  // Numero di matrici nell'array

// static void write_matrix(int*** lattice_history, double*** chemical_history, int num_matrices, int lattice_size);

static void write_matrix(int*** lattice_history, double*** chemical_history, int num_matrices, int lattice_size) {
    FILE *file_lattice, *file_chemical;
    file_lattice = fopen("lattice.txt", "w");
    file_chemical = fopen("chemical.txt", "w");

    if (file_lattice == NULL || file_chemical == NULL) {
        printf("Errore nell'apertura del file\n");
        return;
    }

    // Scrive tutte le matrici di lattice_history
    for (int t = 0; t < num_matrices; t++) {
        fprintf(file_lattice, "Matrice %d:\n", t);
        for (int i = 0; i < lattice_size; i++) {
            for (int j = 0; j < lattice_size; j++) {
                fprintf(file_lattice, "%d ", lattice_history[t][i][j]);
            }
            fprintf(file_lattice, "\n");
        }
        fprintf(file_lattice, "\n");
    }

    // Scrive tutte le matrici di chemical_history
    for (int t = 0; t < num_matrices; t++) {
        fprintf(file_chemical, "Matrice %d:\n", t);
        for (int i = 0; i < lattice_size; i++) {
            for (int j = 0; j < lattice_size; j++) {
                fprintf(file_chemical, "%.8e ", chemical_history[t][i][j]);
            }
            fprintf(file_chemical, "\n");
        }
        fprintf(file_chemical, "\n");
    }

    fclose(file_lattice);
    fclose(file_chemical);
    printf("Array di matrici scritte correttamente su file.\n");
}

#endif
