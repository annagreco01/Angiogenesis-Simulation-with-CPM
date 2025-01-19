// Main che permette di eseguire direttamente una simulazione di visualize.cpp 
//leggendo direttamente i file di input già generati o di eseguire prima la simulazione di cpm.c 
// e poi la visualizzazione in visualize.cpp
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main() {
    char choice[10];
    char simulationType[10];

    printf("Scegli l'operazione da eseguire:\n");
    printf("1. Esegui la simulazione e poi la visualizzazione\n");
    printf("2. Esegui solo la visualizzazione da file pre esistenti\n");
    printf("Inserisci la tua scelta: ");
    scanf("%s", choice);

    if (strcmp(choice, "1") == 0) {
        printf("Scegli il tipo di simulazione:\n");
        printf("1. Simulazione con CUDA\n");
        printf("2. Simulazione sequenziale\n");
        printf("Inserisci la tua scelta: ");
        scanf("%s", simulationType);

        if (strcmp(simulationType, "1") == 0) {
            printf("Scegli il tipo di matrice (scrivi 'dense' per dense, lascia vuoto per sparse): ");
            scanf("%s", choice);

            if (strcmp(choice, "dense") == 0) {
                printf("Esecuzione di cpm.out con CUDA e matrici dense...\n");
                if (system("./cpm_cuda.out dense") != 0) {
                    fprintf(stderr, "Errore durante l'esecuzione di cpm.out cuda dense\n");
                    return 1;
                }
            } else {
                printf("Esecuzione di cpm.out con CUDA e matrici sparse...\n");
                if (system("./cpm_cuda.out") != 0) {
                    fprintf(stderr, "Errore durante l'esecuzione di cpm.out cuda\n");
                    return 1;
                }
            }
        } else if (strcmp(simulationType, "2") == 0) {
            printf("Scegli il tipo di matrice (scrivi 'dense' per dense, lascia vuoto per sparse): ");
            scanf("%s", choice);

            if (strcmp(choice, "dense") == 0) {
                printf("Esecuzione di cpm.out sequenziale con matrici dense...\n");
                if (system("./cpm.out dense") != 0) {
                    fprintf(stderr, "Errore durante l'esecuzione di cpm.out dense\n");
                    return 1;
                }
            } else {
                printf("Esecuzione di cpm.out sequenziale con matrici sparse...\n");
                if (system("./cpm.out") != 0) {
                    fprintf(stderr, "Errore durante l'esecuzione di cpm.out\n");
                    return 1;
                }
            }
        } else {
            printf("Scelta non valida. Esecuzione terminata.\n");
            return 1;
        }

        printf("Esecuzione di visualize.out...\n");
        if (system("./visualize.out") != 0) {
            fprintf(stderr, "Errore durante l'esecuzione di visualize.out\n");
            return 1;
        }

        printf("Esecuzione completata con successo.\n");
    } else if (strcmp(choice, "2") == 0) {
        // Solo visualizzazione
        printf("Esecuzione di visualize.out...\n");
        if (system("./visualize.out") != 0) {
            fprintf(stderr, "Errore durante l'esecuzione di visualize.out\n");
            return 1;
        }

        printf("Esecuzione completata con successo.\n");
    } else {
        printf("Scelta non valida. Esecuzione terminata.\n");
    }

    return 0;
}
