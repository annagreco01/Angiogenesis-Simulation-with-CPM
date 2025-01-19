# Angiogenesis-Simulation-with-CPM

# Repository Tesi di Laurea Triennale in Informatica

## 📜 Descrizione

Questo repository contiene i materiali relativi alla mia **tesi di laurea triennale in Informatica**, discussa presso l'**Università degli Studi di Napoli Parthenope** il **19 dicembre 2024**. Include:  
- Il codice sorgente sviluppato per il progetto.  
- La tesi scritta in formato PDF (`thesis.pdf`) presente nella radice del repository.  

### Requisiti
Per eseguire correttamente il programma, è necessario installare le seguenti librerie:  
1. **SuiteSparse**: Utilizzata per la gestione delle matrici sparse tramite **CSparse**.  
   - [SuiteSparse GitHub](https://github.com/DrTimothyAldenDavis/SuiteSparse)  
2. **cuDSS**: Una libreria per la gestione delle matrici sparse su GPU, non ancora inclusa ufficialmente nel toolkit di NVIDIA.  
   - Per installarla, fai riferimento alla documentazione di NVIDIA.  
   - [cuDSS Download](https://developer.nvidia.com/cudss-downloads)  
3. **LAPACK**: Necessaria per il supporto alle matrici dense.  
   - [LAPACK GitHub](https://github.com/Reference-LAPACK/lapack)  
   - Su sistemi basati su Linux, può essere installata con il comando:  
     ```bash
     sudo apt install liblapack-dev
     ```

---

## 📂 Contenuto del Repository

- `cpm_with_visualize/`: Contiene il codice sorgente del progetto.  
- `thesis.pdf`: La tesi scritta in formato PDF.

---

## ⚙️ Installazione

1. Clona il repository:  
   ```bash
   git clone https://github.com/annagreco01/Angiogenesis-Simulation-with-CPM.git
   cd Angiogenesis-Simulation-with-CPM
   ```
2. Installa le dipendenze richieste (SuiteSparse, cuDSS, LAPACK).  
3. **Adatta il file `Makefile` se necessario**:  
   Il `Makefile` fornito potrebbe dover essere modificato per soddisfare le tue esigenze specifiche, in particolare per:  
   - Percorsi delle librerie installate.  
   - Opzioni di compilazione personalizzate.  
   - Compatibilità con il tuo ambiente di sviluppo.

4. Compila il progetto con il comando:  
   ```bash
   make
   ```

---

# Bachelor's Thesis Repository in Computer Science

## 📜 Description

This repository contains the materials for my **bachelor's thesis in Computer Science**, discussed at **Università degli Studi di Napoli Parthenope** on **December 19, 2024**. It includes:  
- The source code developed for the project.  
- The thesis document in PDF format (`thesis.pdf`) located at the root of the repository.  

### Requirements
To run the program correctly, you need to install the following libraries:  
1. **SuiteSparse**: Used for sparse matrix operations via **CSparse**.  
   - [SuiteSparse GitHub](https://github.com/DrTimothyAldenDavis/SuiteSparse)  
2. **cuDSS**: A library for sparse matrix operations on GPU, not yet officially included in NVIDIA's toolkit.  
   - Refer to NVIDIA documentation.  
   - [cuDSS Download](https://developer.nvidia.com/cudss-downloads)  
3. **LAPACK**: Required for dense matrix support.  
   - [LAPACK GitHub](https://github.com/Reference-LAPACK/lapack)  
   - On Linux-based systems, it can be installed using:  
     ```bash
     sudo apt install liblapack-dev
     ```

---

## 📂 Repository Content

- `cpm_with_visualize/`: Contains the project source code.  
- `thesis.pdf`: The thesis document in PDF format.

---

## ⚙️ Installation

1. Clone the repository:  
   ```bash
   git clone https://github.com/annagreco01/Angiogenesis-Simulation-with-CPM.git
   cd Angiogenesis-Simulation-with-CPM
   ```
2. Install the required dependencies (SuiteSparse, cuDSS, LAPACK).  
3. **Adjust the `Makefile` if necessary**:  
   The provided `Makefile` might need modifications to meet your specific requirements, especially for:  
   - Paths to the installed libraries.  
   - Custom compilation options.  
   - Compatibility with your development environment.

4. Build the project with the command:  
   ```bash
   make
   ```
