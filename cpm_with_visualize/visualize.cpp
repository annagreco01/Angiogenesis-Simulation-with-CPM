#include "visualize.hpp"

int main() {
    if (visualizeInit() != 0) {
        std::cerr << "Errore durante l'inizializzazione della visualizzazione." << std::endl;
        return -1;
    } else {
        std::cout << "Animazione salvata come cpm_simulation_opencv.mp4\n";
    }
    return 0;
}
