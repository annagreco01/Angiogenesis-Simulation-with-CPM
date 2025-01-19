#ifndef CALC_TIME_H
#define CALC_TIME_H

#include <stdio.h>
#include <time.h>
#include <math.h>

// Verifica se il codice è compilato in C o C++
#ifdef __cplusplus
extern "C" { // extern serve per evitare problemi con il linkage del C++, quindi specifica che il codice è in C 
#endif

// Struct per memorizzare i tempi
typedef struct {
    time_t start;
    time_t last_check;
} TimeTracker;

// Funzioni di gestione del tempo
void calc_timeStart();
void calc_timeCheck();
void calc_timePrint();

#ifdef __cplusplus
}
#endif

#endif
