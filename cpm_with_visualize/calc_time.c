#include "calc_time.h"

char total_buffer[20]="\0";
char last_buffer[20]= "\0";
TimeTracker tracker;
time_t now;

// Funzione per inizializzare il tracker
void initializeTracker(TimeTracker *tracker) {
    time(&tracker->start);
    tracker->last_check = tracker->start;
}

// Funzione per formattare il calc_time in GG-HH:MM:SS.sss
void formatDuration(double seconds, char *buffer) {
    int days = (int)(seconds / (24 * 3600));
    seconds = fmod(seconds, 24 * 3600);
    int hours = (int)(seconds / 3600);
    seconds = fmod(seconds, 3600);
    int minutes = (int)(seconds / 60);
    seconds = fmod(seconds, 60);
    int secs = (int)seconds;
    int millisecs = (int)((seconds - secs) * 1000);
    
    sprintf(buffer, "%02d-%02d:%02d:%02d.%03d", days, hours, minutes, secs, millisecs);
}

// Funzione per controllare e stampare il calc_time trascorso
void checkTime(TimeTracker *tracker) {
    time(&now);
    
    double elapsed_total = difftime(now, tracker->start);
    double elapsed_since_last = difftime(now, tracker->last_check);
    
    formatDuration(elapsed_total, total_buffer);
    formatDuration(elapsed_since_last, last_buffer);
    
    tracker->last_check = now;
}

void calc_timeStart() {
    initializeTracker(&tracker);
}

void calc_timeCheck() {
    checkTime(&tracker);
}

void calc_timePrint() {
    struct tm *local_time;
    char current_time_buffer[100]="\0";
    local_time = localtime(&now);
    // Formatta la data
    strftime(current_time_buffer, sizeof(current_time_buffer), 
             "%d-%m-%Y %H:%M:%S", local_time);

    printf("Data ora: %s - calc_time totale: %s\n", current_time_buffer, total_buffer);
    printf("calc_time dall'ultimo controllo: %s\n", last_buffer);
}