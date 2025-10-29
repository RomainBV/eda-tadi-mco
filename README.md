# Detection and Localization of Leaks on TADI - MCO
author : Romain B
## Overview
New-generation sensors (Rogue board and Infineon microphones) are more sensitive than Vesper in the ultrasonic domain, making them better suited for gas leak detection.  
This new hardware integration required an update of the machine-learning-based leak detection model.  
An additional "Expert" model, based on acoustic indicators and dedicated filtering, is also evaluated.

This repository provides the tools for:
- **Feature computation**
- **Bandleq processing**
- **Expert leak detection**
- **Machine Learning (ML) leak detection**

These tools are designed to evaluate sensor performance from **in-situ deployments**.

---

## Test Campaigns
Evaluation is conducted through several experiments, each documented with reports and datasets:

- **10/10/2025** – Caltech – [Documentation link](https://docs.google.com/presentation/d/1cS_0U8yQzuMo1AWq5dP4kX8BFjXrpakc/edit?slide=id.p1#slide=id.p1)
- *(Add more test dates and documentation links as needed)*

---

## Environments and Dependencies
Two separate virtual environments (**venv**) are required because the `signal` version of `datasets-analysis` is incompatible with `waveleaks`.

- **Feature extraction environment:**  
  Install dependencies using  
  ```bash
  pip install -r requirements_features.txt
  ```
- **Leak detection environment:**
  Install dependencies using
  ```bash
  pip install -r requirements_venv_leak_detection_model.txt
  ```

### 1. Update annotations
Add client annotation data to:  
`TADI - All datasets analyses.xlsx`

---

### 2. Retrieve server data
From [TADI Cerebro Server](https://tadi.cerebro.wavely.fr/):  
Export **sound levels** and **leak detection data** via  
**More → Export CSV** from affiliated Dasboards, then insert them into:  
`TADI_MCO-2025_Sound-levels_Leak-detection.xls
### 3. Prepare data environment:
Create and activate a virtual environment, then install dependencies:
```bash
pip install -r requirements_features.txt
```
### 4. Download data:
```bash
mc cp --recursive data2/... data/MAGNETO/<sensor_ID> <local folder path>
```
### 5. Preporcessing
```bash
streamlit run preprocessing.py
```
This opens a Streamlit interface for raw data preprocessing.
### 6. Data annotation:
Annotate raw data using tools like Audacity, following
`TADI - All datasets analyses.xlsx`
Refer to the Streamlit interface for detailed instructions on how to perform annotations.

### 7. Feature computation and Expert model:
```bash
streamlit run processing.py
```
This computes labeled features and runs Expert Leak Detection.
### 8. Machine Learning Leak Detection:
Activate the ML environment and run:
```bash
streamlit run ml_leak_predict_from_onnx.py
```
The results are automatically aggregated into results/<sensor_ID>/features.h5.

### 9. Postprocessing and Visualization:
Reactivate the feature environment and run:
```bash
streamlit run postprocessing.py
```
Opens a Streamlit dashboard for visualizing:
- Features
- Expert Leak Detection
- ML Leak Detection
- Spectrograms over time

Enables precise temporal comparison of model performance.

### 10. Leak Localization Analysis:
Run:
```bash
python Leak_localisation_analysis.py
```
Generates statistical tables and figures for final reporting.

---
## Repository Structure
```bash 
data/
  └── MAGNETO/
      └── <sensor_ID>/
results/
  └── <sensor_ID>/
      └── features.h5
scripts/
  ├── preprocessing.py
  ├── processing.py
  ├── ml_leak_predict_from_onnx.py
  ├── postprocessing.py
  └── Leak_localisation_analysis.py
```
