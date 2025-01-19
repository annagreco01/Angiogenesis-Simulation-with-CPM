#ifndef VISUALIZE_HPP
#define VISUALIZE_HPP
#include <opencv2/opencv.hpp>
#include <opencv2/freetype.hpp> // Include il modulo FreeType per il testo in OpenCV
#include <iostream>
#include <fstream>
#include <vector>
#include <unordered_map>

// Definizione dei parametri principali
#define LATTICE_SIZE 200
#define NUM_CELLS 10
#define TARGET_VOLUME 500
#define NUM_MCS 100
#define T 20.0  // Temperatura per l'algoritmo di Metropolis
const int upscale_factor = 5; // Modifica questo fattore per aumentare la risoluzione (2x, 3x, ecc.)

std::string frame_name; // Nome del frame per la visualizzazione

// Strutture per memorizzare tutte le matrici di lattice e chimiche
std::unordered_map<std::string, std::vector<cv::Mat>> latticeFramesMap;
std::unordered_map<std::string, std::vector<cv::Mat>> chemicalFramesMap;

// Dichiarazione delle funzioni (assicurati che siano implementate correttamente nel tuo codice)
// Funzione per visualizzare le celle sulla matrice lattice
void visualize_cells(cv::Mat &lattice, cv::Mat &display, int upscale_size);
// Funzione per leggere la matrice da un file, specificando se è chimica o meno
void loadAllFrames(const std::string& filename, bool isChemical, int frame_size);
// Funzione per leggere la matrice lattice da un file
cv::Mat readLatticeMatrix(const std::string& filename, int frame_index);
// Funzione per leggere e processare i dati chimici come CV_32F e convertirli in CV_8UC1
cv::Mat readAndProcessChemicalMatrix(const std::string& filename, int frame_index);
// Funzione per disegnare le coordinate esterne all'immagine
void drawCoordinatesOutside(cv::Mat &frame, int offset_x, int offset_y, int image_size, int step, bool is_x_axis, int scale_factor);
// Funzione per creare una legenda verticale (color bar)
cv::Mat createColorBar(int height, int width, int num_ticks, cv::ColormapTypes colormap);
// Funzione per aggiungere etichette alla barra dei colori
void addColorBarLabels(cv::Mat& frame_image, int x, int y, int bar_width, int bar_height, int num_ticks, double min_value, double max_value, cv::Ptr<cv::freetype::FreeType2> ft2, int scale_factor);
// Funzione per visualizzare i dati
int visualizeInit(); 

void loadAllFrames(const std::string& filename, bool isChemical, int frame_size) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error opening file: " << filename << std::endl;
        return;
    }

    std::vector<cv::Mat> frames;
    std::string line;
    int current_frame = -1;
    cv::Mat matrix = cv::Mat::zeros(frame_size, frame_size, isChemical ? CV_32F : CV_8UC1);
    int row = 0;

    while (getline(file, line)) {
        if (line.find("Matrice") != std::string::npos) {
            if (current_frame >= 0) {
                frames.push_back(matrix.clone());
            }
            current_frame++;
            row = 0;
            matrix = cv::Mat::zeros(frame_size, frame_size, isChemical ? CV_32F : CV_8UC1);
            continue;
        }

        if (current_frame >= 0) {
            std::stringstream ss(line);
            int col = 0;
            if (isChemical) {
                float value;
                while (ss >> value) {
                    if (col < frame_size && row < frame_size) {
                        matrix.at<float>(row, col) = value;
                        col++;
                    }
                }
            } else {
                int value;
                while (ss >> value) {
                    if (col < frame_size && row < frame_size) {
                        matrix.at<uchar>(row, col) = static_cast<uchar>(value);
                        col++;
                    }
                }
            }
            row++;
        }
    }

    if (current_frame >= 0) {
        frames.push_back(matrix.clone());
    }

    if (isChemical) {
        chemicalFramesMap[filename] = frames;
    } else {
        latticeFramesMap[filename] = frames;
    }

    file.close();
}

cv::Mat readLatticeMatrix(const std::string& filename, int frame_index) {
    if (latticeFramesMap.find(filename) == latticeFramesMap.end()) {
        loadAllFrames(filename, false, LATTICE_SIZE);
    }
    
    if (frame_index >= 0 && static_cast<size_t>(frame_index) < latticeFramesMap[filename].size()) {
        return latticeFramesMap[filename][frame_index];
    } else {
        std::cerr << "Frame index out of bounds: " << frame_index << std::endl;
        return cv::Mat::zeros(LATTICE_SIZE, LATTICE_SIZE, CV_8UC1);
    }
}

cv::Mat readAndProcessChemicalMatrix(const std::string& filename, int frame_index) {
    if (chemicalFramesMap.find(filename) == chemicalFramesMap.end()) {
        loadAllFrames(filename, true, LATTICE_SIZE);
    }
    
    if (frame_index >= 0 && static_cast<size_t>(frame_index) < chemicalFramesMap[filename].size()) {
        cv::Mat matrix = chemicalFramesMap[filename][frame_index];

        // Normalizza e converte per la visualizzazione
        cv::Mat normalized;
        cv::normalize(matrix, normalized, 0, 255, cv::NORM_MINMAX);
        normalized.convertTo(normalized, CV_8UC1);
        return normalized;
    } else {
        std::cerr << "Frame index out of bounds: " << frame_index << std::endl;
        return cv::Mat::zeros(LATTICE_SIZE, LATTICE_SIZE, CV_8UC1);
    }
}


void visualize_cells(cv::Mat &lattice, cv::Mat &display, int upscale_size) {
    int rows = lattice.rows;
    int cols = lattice.cols;
    std::vector<cv::Vec3b> cell_colors(NUM_CELLS + 1);
    cv::RNG rng(12345);

    // Creazione di colori distinti per ciascuna cella
    for (int i = 1; i <= NUM_CELLS; ++i) {        
        int b = rng.uniform(0, 256);
        int g = rng.uniform(0, 256);
        int r = rng.uniform(0, 256);
        cell_colors[i] = cv::Vec3b(b, g, r);
    }
    
    // Crea una versione ingrandita per disegno, serve per disegnare i contorni sottili delle celle
    int multiply = 2;
    int scaled_upscale_factor = upscale_factor * multiply; // Aumentiamo il fattore di scala per contorni sottili
    cv::Mat large_display(rows * scaled_upscale_factor, cols * scaled_upscale_factor, CV_8UC3, cv::Scalar(255, 255, 255)); // Immagine ingrandita

    // Disegna le celle riempiendole
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            uchar value = lattice.at<uchar>(i, j);
            if (value > 0) {
                cv::Rect cell_rect(j * scaled_upscale_factor, i * scaled_upscale_factor, scaled_upscale_factor, scaled_upscale_factor);
                cv::rectangle(large_display, cell_rect, cell_colors[value], -1);
            }
        }
    }

    // Disegna i contorni sottili
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            uchar value = lattice.at<uchar>(i, j);
            if (value > 0) {
                bool is_boundary = false;
                for (int di = -1; di <= 1; ++di) {
                    for (int dj = -1; dj <= 1; ++dj) {
                        if (di == 0 && dj == 0) continue;
                        int ni = (i + di + rows) % rows;
                        int nj = (j + dj + cols) % cols;

                        if (lattice.at<uchar>(ni, nj) != value) {
                            is_boundary = true;
                            break;
                        }
                    }
                    if (is_boundary) {
                        // Disegna i bordi con linee sottili solo sul bordo del pixel
                        int x = j * scaled_upscale_factor;
                        int y = i * scaled_upscale_factor;
                       cv::rectangle(large_display, cv::Rect(x, y, scaled_upscale_factor, scaled_upscale_factor), cv::Scalar(0, 0, 0), 1);
                        break;
                    }
                }
            }
        }
    }

    // Ridimensiona l'immagine ingrandita alla dimensione originale con antialiasing
    //cv::imwrite(frame_name + ".png", large_display);
    cv::resize(large_display, display, cv::Size(upscale_size, upscale_size), 0, 0, cv::INTER_AREA);
}

void drawCoordinatesOutside(cv::Mat &frame, int offset_x, int offset_y, int image_size, int step, bool is_x_axis, int scale_factor) {
    cv::Ptr<cv::freetype::FreeType2> ft2_helvetica_roman = cv::freetype::createFreeType2();
    ft2_helvetica_roman->loadFontData("fonts/helvetica-neue/HelveticaNeueRoman.otf", 0); // Carica il font

    // Moltiplica per un fattore di scala maggiore per rendere il testo e le tacche più visibili
    int base_font_height = 10; // Altezza di base del font
    int font_height = std::max(static_cast<int>(base_font_height * (scale_factor / 50.0)), 15); // Scala di più il font e imposta un minimo più alto
    cv::Scalar font_color(0, 0, 0); // Colore del testo (nero)
    int base_tick_length = 5; // Lunghezza di base delle tacche
    int tick_length = std::max(static_cast<int>(base_tick_length * (scale_factor / 50.0)), 3); // Scala di più le tacche e imposta un minimo più alto
    int base_label_offset = 8; // Offset di base delle etichette
    int label_offset = std::max(static_cast<int>(base_label_offset * (scale_factor / 50.0)), 10); // Scala l'offset di più e imposta un minimo più alto

    // Disegna le coordinate sull'asse X (sopra e sotto l'immagine)
    if (is_x_axis) {
        for (int x = step; x <= image_size / scale_factor; x += step) {
            std::string label = std::to_string(x);
            cv::Size textSize = ft2_helvetica_roman->getTextSize(label, font_height, -1, nullptr);

            // Posizione centrata sulla coordinata x scalata
            int text_x = offset_x + x * scale_factor - textSize.width / 2;
            int text_y_bottom = offset_y + image_size + textSize.height + label_offset; // Aggiungi offset per spostare l'etichetta

            // Disegna il testo e le tacche
            if (text_x > 0 && text_x < frame.cols && text_y_bottom > 0 && text_y_bottom < frame.rows) {
                ft2_helvetica_roman->putText(frame, label, cv::Point(text_x, text_y_bottom), font_height, font_color, cv::FILLED, cv::LINE_AA, true);
                cv::line(frame, cv::Point(offset_x + x * scale_factor, offset_y + image_size), 
                         cv::Point(offset_x + x * scale_factor, offset_y + image_size + tick_length), font_color, 1);
            }
        }

        // Aggiunge l'etichetta 'X' al centro dell'asse X
        std::string x_label = "X";
        cv::Size x_label_size = ft2_helvetica_roman->getTextSize(x_label, font_height, -1, nullptr);
        int x_label_x = offset_x + image_size / 2 - x_label_size.width / 2;
        int x_label_y = offset_y - 10; // Posizionato sopra l'immagine
        if (x_label_x > 0 && x_label_x < frame.cols && x_label_y > 0 && x_label_y < frame.rows) {
            ft2_helvetica_roman->putText(frame, x_label, cv::Point(x_label_x, x_label_y), font_height, font_color, cv::FILLED, cv::LINE_AA, true);
        }
    }
    // Disegna le coordinate sull'asse Y (a sinistra e a destra dell'immagine)
    else {
        for (int y = step; y <= image_size / scale_factor; y += step) {
            std::string label = std::to_string(y);
            cv::Size textSize = ft2_helvetica_roman->getTextSize(label, font_height, -1, nullptr);

            int text_x_left = offset_x - textSize.width - label_offset; // Aggiungi offset per spostare l'etichetta a sinistra
            int text_y = offset_y + y * scale_factor + textSize.height / 2;

            // Disegna il testo e le tacche
            if (text_x_left > 0 && text_x_left < frame.cols && text_y > 0 && text_y < frame.rows) {
                ft2_helvetica_roman->putText(frame, label, cv::Point(text_x_left, text_y), font_height, font_color, cv::FILLED, cv::LINE_AA, true);
                cv::line(frame, cv::Point(offset_x - tick_length, offset_y + y * scale_factor), 
                         cv::Point(offset_x, offset_y + y * scale_factor), font_color, 1);
            }
        }

        // Aggiunge l'etichetta 'Y' al centro dell'asse Y
        std::string y_label = "Y";
        cv::Size y_label_size = ft2_helvetica_roman->getTextSize(y_label, font_height, -1, nullptr);
        int y_label_x = offset_x + image_size + 10; // Posizionato a destra dell'immagine
        int y_label_y = offset_y + image_size / 2 + y_label_size.height / 2;
        if (y_label_x > 0 && y_label_x < frame.cols && y_label_y > 0 && y_label_y < frame.rows) {
            ft2_helvetica_roman->putText(frame, y_label, cv::Point(y_label_x, y_label_y), font_height, font_color, cv::FILLED, cv::LINE_AA, true);
        }
    }
}

// Funzione per creare una legenda verticale (color bar)
cv::Mat createColorBar(int height, int width, int num_ticks, cv::ColormapTypes colormap) {
    // Crea un'immagine a gradiente verticale (da 0 a 255)
    cv::Mat gradient(height, 1, CV_8UC1);
    for (int i = 0; i < height; ++i) {
        gradient.at<uchar>(i, 0) = static_cast<uchar>((255.0 * i) / height);
    }
    // Inverti la mappa di colori per avere il bianco in alto e nero in basso
    cv::flip(gradient, gradient, 0);
    cv::Mat color_bar;
    cv::applyColorMap(gradient, color_bar, colormap);

    // Ridimensiona per la larghezza desiderata
    cv::resize(color_bar, color_bar, cv::Size(width, height));
    cv::rectangle(color_bar, cv::Point(0, 0), cv::Point(color_bar.cols - 1, color_bar.rows + 3), cv::Scalar(0, 0, 0), num_ticks);

    return color_bar;
}

void addColorBarLabels(cv::Mat& frame_image, int x, int y, int bar_width, int bar_height, int num_ticks, double min_value, double max_value, cv::Ptr<cv::freetype::FreeType2> ft2, int scale_factor) {
    // Scala il font_height e la lunghezza delle tacche in base al scale_factor
    int base_font_height = 12; // Altezza di base del font
    int font_height = std::max(static_cast<int>(base_font_height * (scale_factor / 50.0)), 15); // Scala di più il font e imposta un minimo più alto
    cv::Scalar font_color(0, 0, 0); // Colore del testo (nero)
    int base_tick_length = 5; // Lunghezza di base delle tacche
    int tick_length = std::max(static_cast<int>(base_tick_length * (scale_factor / 50.0)), 3); // Scala di più le tacche e imposta un minimo più alto

    // Disegna le etichette e le tacche lungo la barra
    for (int i = 0; i <= num_ticks; ++i) {
        int y_tick = y + bar_height - (i * bar_height / num_ticks) - 1; // Posizione y per l'etichetta
        double value = min_value + i * (max_value - min_value) / num_ticks; // Calcolo del valore

        // Disegna le etichette a destra della color bar
        std::string label = cv::format("%.1f", value);
        cv::Size text_size = ft2->getTextSize(label, font_height, -1, nullptr);
        int x_label = x + bar_width + 15; // Distanza dalla color bar
        ft2->putText(frame_image, label, cv::Point(x_label, y_tick + text_size.height / 2), font_height, font_color, cv::FILLED, cv::LINE_AA, true);

        // Disegna le tacche allineate al bordo destro della color bar
        cv::line(frame_image, cv::Point(x + bar_width, y_tick), cv::Point(x + bar_width + tick_length, y_tick), font_color, 1);
    }
}


int visualizeInit() {
    // Adegua i percorsi dei file come necessario
    std::string latticeFile = "lattice.txt";
    std::string chemicalFile = "chemical.txt";

    int num_frames = 11; // Numero di frame effettivi salvati nei file (da 0 a 10)
    int desired_fps = 10; // Frame rate desiderato per una riproduzione fluida

     // Carica il modulo FreeType per usare font personalizzati
    cv::Ptr<cv::freetype::FreeType2> ft2 = cv::freetype::createFreeType2();
    ft2->loadFontData("fonts/helvetica-neue/HelveticaNeueMedium.otf", 0); // Carica il font
    cv::Ptr<cv::freetype::FreeType2> ft2_helvetica_light = cv::freetype::createFreeType2();
    ft2_helvetica_light->loadFontData("fonts/helveticaneue-2/Light.ttf", 0); // Carica il font

    // Definizione delle dimensioni con padding e spazi per le coordinate
    int upscale_size = LATTICE_SIZE * upscale_factor;

    // Aumenta il padding per avere più spazio ai lati del video
    int side_padding = 100; // Aumenta il padding per maggiore spazio
    int top_padding = 50; // Aggiunge spazio sopra
    int bottom_padding = 50; // Aggiunge spazio sotto
    int title_padding = 10; // Margine sopra i titoli

    int title_height = 50; // Altezza per i titoli
    int footer_height = 30; // Altezza per il testo del frame in basso
    int total_width = 2 * upscale_size + 3 * side_padding; // Spazio per le due immagini e i margini
    int total_height = upscale_size + title_height + footer_height + top_padding + bottom_padding; // Spazio per l'immagine, i titoli e i margini superiori/inferiori

    cv::Size video_size(total_width, total_height);
    //fourcc('H','2','6','4'); // Codec per il formato H.264
    //fourcc('M','P','4','V'); // Codec per il formato MP4
    cv::VideoWriter video("cpm_simulation_opencv.mp4", cv::VideoWriter::fourcc('H','2','6','4'), desired_fps, video_size, true);
    if (!video.isOpened()) {
        std::cout << "Errore: Non è stato possibile aprire il video writer." << std::endl;
        return -1;
    }

    // Prima del ciclo for, carica tutti i frame
    loadAllFrames("lattice.txt", false, LATTICE_SIZE); // Carica tutte le matrici lattice
    loadAllFrames("chemical.txt", true, LATTICE_SIZE); // Carica tutte le matrici chimiche

    for (int frame = 0; frame < num_frames; ++frame) {
        cv::Mat lattice = readLatticeMatrix(latticeFile, frame);
        cv::Mat chemical = readAndProcessChemicalMatrix(chemicalFile, frame);

        // Preparazione delle immagini di visualizzazione
        cv::Mat display_lattice(upscale_size, upscale_size, CV_8UC3, cv::Scalar(255, 255, 255));
        cv::Mat display_chemical(upscale_size, upscale_size, CV_8UC3, cv::Scalar(0, 0, 0));

        //frame_name = "Frame " + std::to_string(frame);
        visualize_cells(lattice, display_lattice, upscale_size); 
        cv::applyColorMap(chemical, display_chemical, cv::COLORMAP_HOT);

        // Ridimensionamento per una risoluzione più alta
        cv::resize(display_lattice, display_lattice, cv::Size(upscale_size, upscale_size), 0, 0, cv::INTER_CUBIC);
        cv::resize(display_chemical, display_chemical, cv::Size(upscale_size, upscale_size), 0, 0, cv::INTER_CUBIC);

        // Combinazione delle immagini con padding attorno ai bordi
        cv::Mat frame_image(total_height, total_width, CV_8UC3, cv::Scalar(255, 255, 255)); // Sfondo bianco

        // Aggiungi i titoli centrati sopra ciascuna finestra
        std::string title_lattice = "Cell Configuration";
        std::string title_chemical = "Chemical Concentration";
        // Definizione delle proprietà per il testo
        int font_height = 30; // Altezza del font
        cv::Scalar font_color(0, 0, 0); // Colore del testo (nero)

        // Calcolo della posizione centrata per il titolo "Cell Configuration"
        cv::Size text_size_lattice = ft2->getTextSize(title_lattice, font_height, -1, nullptr);
        int text_x_lattice = side_padding + (upscale_size - text_size_lattice.width) / 2; // Posizione centrata per il titolo
        int text_y_lattice = top_padding + title_padding; // Aggiungi il margine di padding sopra il titolo

        // Calcolo della posizione centrata per il titolo "Chemical Concentration"
        cv::Size text_size_chemical = ft2->getTextSize(title_chemical, font_height, -1, nullptr);
        int text_x_chemical = 2 * side_padding + upscale_size + (upscale_size - text_size_chemical.width) / 2;
        int text_y_chemical = top_padding + title_padding; // Aggiungi il margine di padding sopra il titolo

        // Crea la legenda dei colori
        int color_bar_height = upscale_size; // Altezza uguale alla matrice
        int color_bar_width = 20; // Larghezza della barra dei colori
        cv::Mat color_bar = createColorBar(color_bar_height, color_bar_width, 2, cv::COLORMAP_HOT);

        // Posiziona la legenda a destra della matrice di concentrazione chimica
        int color_bar_x = 2 * side_padding + 2 * upscale_size + 30; // Posizione x
        int color_bar_y = top_padding + title_height; // Posizione y, faccio
        // Aggiungi le etichette alla barra dei colori
        addColorBarLabels(frame_image, color_bar_x, color_bar_y + 2, color_bar_width, color_bar_height - 3, 5, -1.0, 1.0, ft2_helvetica_light, upscale_factor);

        // Copia la legenda nel frame finale
        color_bar.copyTo(frame_image(cv::Rect(color_bar_x, color_bar_y, color_bar_width, color_bar_height)));

        // Scrive i titoli
        ft2->putText(frame_image, title_lattice, cv::Point(text_x_lattice, text_y_lattice), font_height, font_color, cv::FILLED, cv::LINE_AA, true);
        ft2->putText(frame_image, title_chemical, cv::Point(text_x_chemical, text_y_chemical), font_height, font_color, cv::FILLED, cv::LINE_AA, true);

        // Copia le immagini con padding attorno ai bordi
        display_lattice.copyTo(frame_image(cv::Rect(side_padding, top_padding + title_height, upscale_size, upscale_size)));
        display_chemical.copyTo(frame_image(cv::Rect(2 * side_padding + upscale_size, top_padding + title_height, upscale_size, upscale_size)));

        // Aggiungi un contorno attorno alla matrice lattice
        cv::rectangle(frame_image, 
                      cv::Point(side_padding, top_padding + title_height), 
                      cv::Point(side_padding + upscale_size, top_padding + title_height + upscale_size), 
                      cv::Scalar(0, 0, 0), 2);

        // Aggiungi un contorno attorno alla matrice chemical
        cv::rectangle(frame_image, 
                      cv::Point(2 * side_padding + upscale_size, top_padding + title_height), 
                      cv::Point(2 * side_padding + 2 * upscale_size, top_padding + title_height + upscale_size), 
                      cv::Scalar(0, 0, 0), 2);

        // Disegna le coordinate esternamente
        drawCoordinatesOutside(frame_image, side_padding, top_padding + title_height, upscale_size, 20, true, upscale_factor); // Coordinate per l'asse X di lattice
        drawCoordinatesOutside(frame_image, side_padding, top_padding + title_height, upscale_size, 20, false, upscale_factor); // Coordinate per l'asse Y di lattice
        drawCoordinatesOutside(frame_image, 2 * side_padding + upscale_size, top_padding + title_height, upscale_size, 20, true, upscale_factor); // Coordinate per l'asse X di chemical
        drawCoordinatesOutside(frame_image, 2 * side_padding + upscale_size, top_padding + title_height, upscale_size, 20, false, upscale_factor); // Coordinate per l'asse Y di chemical

        // Aggiungi il testo in basso per indicare il frame corrente, centrato
        std::string frame_text = "Frame: " + std::to_string(frame + 1) + "/" + std::to_string(num_frames);
        cv::Size text_size_frame = ft2_helvetica_light->getTextSize(frame_text, font_height, -1, nullptr);
        // cv::Size text_size_frame = cv::getTextSize(frame_text, font_face, font_scale, thickness, nullptr);
        int text_x_frame = (total_width - text_size_frame.width) / 2; // Calcolo della posizione x centrata
        int text_y_frame = total_height - (footer_height / 2); // Posizione y per il testo

        //cv::putText(frame_image, frame_text, cv::Point(text_x_frame, text_y_frame), font_face, font_scale, cv::Scalar(0, 0, 0), thickness);
        ft2_helvetica_light->putText(frame_image, frame_text, cv::Point(text_x_frame, text_y_frame), font_height, font_color, cv::FILLED, cv::LINE_AA, true);

        // Scrivi nel video
        video.write(frame_image);

        // Visualizza i frame per il controllo, se necessario
        namedWindow("Chemical Concentration", cv::WINDOW_NORMAL);
        namedWindow("Cell Configuration", cv::WINDOW_NORMAL);
        // Posizionamento delle finestre
        cv::moveWindow("Cell Configuration", 100, 100); // Sposta la finestra "Cell Configuration" alla posizione (100, 100)
        cv::moveWindow("Chemical Concentration", 600, 100); // Sposta la finestra "Chemical Concentration" accanto a "Cell Configuration" (modifica 600 per regolare la distanza)
        // Visualizzazione dei frame
        cv::imshow("Chemical Concentration", display_chemical);
        cv::imshow("Cell Configuration", display_lattice);
        cv::waitKey(1); // Breve pausa per visualizzazione in diretta
    }

    video.release();
    cv::destroyAllWindows();
    return 0;
}

#endif // VISUALIZE_HPP