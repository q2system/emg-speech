# ROLE AND PERSONA
You are an Expert Python Systems Architect and Machine Learning Engineer specializing in Biosignal Processing (EMG/IMU), Real-Time Deep Learning, and Desktop Application Development. You have deep expertise in PyTorch, Digital Signal Processing (DSP), PyQt, and serial communications. Your code is production-ready, highly modular, strictly typed, and adheres to SOLID principles.

# CONTEXT
We are building a subvocal/EMG speech recognition system. The system captures neuromuscular and inertial signals from a wearable device (ESP32 + MPU6050 + 3 EMG channels) and translates them into text. The project is divided into two distinct Python applications: 
1. A Data Acquisition & Annotation Tool.
2. A Model Training & Real-Time Inference Tool.

# PROJECT SPECIFICATIONS

## Application 1: Data Acquisition & Annotation Tool
**Objective:** Record, process, and label 9-channel biosignal data to create a balanced, fixed-length training dataset.
**Hardware Interface:** ESP32 connected via UART (Serial).
**Data Format:** Incoming serial data is a comma-separated string ending in a newline:
`emg1:xxx,emg2:xxx,emg3:xxx,gx:xxx,gy:xxx,gz:xxx,ax:xxx,ay:xxx,az:xxx\n`
*(where 'xxx' is the numerical sensor value)*

**Core Features:**
1. **Class Management:** Allow the user to define a "word" (class). The software creates a corresponding folder for this class.
2. **Recording Workflow:** The user presses a UI button to start recording, speaks/subvocalizes the word, and releases the button to stop.
3. **Advanced Event Detection (Crucial):** The software must NOT just save the raw button-press duration. It must perform real-time DSP (e.g., RMS envelope extraction, thresholding) to detect the *actual* start and end of the muscle activity (the "speech" gesture), ignoring idle time before the button press or after the gesture finishes.
4. **Time Normalization (Crucial):** All extracted events must be padded or truncated to a strict, equal length (e.g., exactly $N$ time steps) to ensure uniform tensor shapes for the neural network.
5. **Storage:** Save the processed, fixed-length 9-channel arrays as `.npy` or `.csv` files inside the respective class folders.

## Application 2: Model Training & Real-Time Inference Tool
**Objective:** Train a sequence model on the collected dataset and deploy it for real-time, continuous speech-to-text transcription.
**Framework:** PyTorch.
**Model Architecture:** LSTM (or BiLSTM/GRU) network for sequence classification.

**Core Features:**
1. **Training Pipeline:** 
   - Custom PyTorch `Dataset` and `DataLoader` to read the fixed-length files from App 1.
   - Standard training loop with cross-entropy loss, Adam optimizer, and validation tracking.
2. **Real-Time Inference Pipeline:**
   - Stream data continuously from the ESP32 UART.
   - Implement a sliding window or continuous event-detection mechanism to feed chunks of data to the model in real-time.
   - Output the recognized words sequentially.
3. **User Interface:**
   - A simple PyQt (PyQt6 or PySide6) application.
   - The UI must display a text field that populates with recognized words in real-time (mimicking standard speech-to-text software).

# STRICT CONSTRAINTS & ARCHITECTURAL RULES
1. **Strict Separation of Concerns (MVC/MVP):** The PyQt GUI code must be 100% decoupled from the core logic (UART reading, DSP, PyTorch inference). Use Qt Signals/Slots, Observer patterns, or separate worker threads (QThread) to ensure the UI never freezes during serial reading or model inference.
2. **No Monolithic Scripts:** Do not write everything in one file. Provide a clear directory structure and modularize the code (e.g., `serial_handler.py`, `dsp_processor.py`, `model.py`, `ui_main.py`).
3. **Real-Time Performance:** Inference in App 2 must be optimized. Avoid blocking the main thread. 
4. **Robustness:** Handle serial disconnects, malformed UART strings, and empty classes gracefully.

# REQUIRED OUTPUT FORMAT
Please structure your response exactly as follows:

## 1. System Architecture & Directory Structure
Briefly explain the data flow and provide the recommended folder/file structure for both applications.

## 2. Application 1 Implementation
Provide the modular Python code for the Data Acquisition Tool. Pay special attention to the DSP event detection and time-normalization logic.

## 3. Application 2 Implementation
Provide the modular Python code for the Training and Inference Tool. Include the PyTorch model definition, the custom Dataset class, and the threaded real-time inference logic.

## 4. PyQt UI Implementation
Provide the strictly separated PyQt code for both applications, demonstrating how the UI communicates with the backend logic without blocking.

## 5. Execution & Dependencies
List the required `pip install` packages and brief instructions on how to run the system.

Take a deep breath, think step-by-step about the DSP requirements for event detection and the threading requirements for the real-time UI, and begin generating the solution.
