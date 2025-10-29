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

The installation is not straightforward because `datasets-analysis` is incompatible with several recent package versions.

The procedure is as follows:

1. **Clone the required Wavely package repositories** (using the specified versions):
   - `wavely.signal` (v0.9.0)
   - `wavely.edge-metadata` (v0.3.5)
   - `wavely.metadata` (v0.3.4)
   - `wavely.datasets-analysis` (master — v1.1.0)

2. **Create a virtual environment** using Python 3.10:  
   `/usr/bin/python3.10 -m venv <path_to_venv>`

3. **Install each repository in editable mode** in the order listed above:  
   `pip install -e .[dev]`

4. **Install additional dependencies**:  
   `pip install --no-deps -r requirements_venv_leak_detection_model.txt`

> ⚠️ Note: Two separate virtual environments (**venvs**) are required because the `wavely.signal` version of `datasets-analysis` is incompatible with `waveleaks`.  
> The second virtual environment should have the `waveleaks` package installed along with the same dependencies:  
> `pip install --no-deps -r requirements_venv_leak_detection_model.txt`


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
mc cp --recursive data2/... data/MAGNETO/<sensor_ID> <Local_folder_path>
```
### 5. Preporcessing
```bash
streamlit run preprocessing.py
```
This opens a Streamlit interface for raw data preprocessing. Launch the computation for each sensor, making sure to select the correct test date.
### 6. Data annotation:
Annotate raw data using tools like Audacity, following
`TADI - All datasets analyses.xlsx`
Refer to the Streamlit interface for detailed instructions on how to perform annotations.

### 7. Feature computation and Expert model:
```bash
streamlit run processing.py
```
This computes labeled features and runs Expert Leak Detection. Launch the computation for each sensor, making sure to select the correct test date. The results are stored into `data/MAGNETO/<sensor_ID/results/<sensor_ID>/features.h5`
### 8. Machine Learning Leak Detection:
Activate the ML environment and run:
```bash
streamlit run ml_leak_predict_from_onnx.py
```
The results are automatically aggregated into `data/MAGNETO/<sensor_ID/results/<sensor_ID>/features.h5`. 

### 9. Postprocessing and Visualization:
Reactivate the feature environment and run:
```bash
streamlit run postprocessing.py
```
Opens a new Streamlit interface for dynamic, time-based visualization of the results.
- Features
- Expert Leak Detection
- ML Leak Detection
- Spectrograms

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
├── preprocessing.py   # Data preprocessing pipeline (Streamlit interface)
├── processing.py      # Feature computation and Expert leak detection (Streamlit interface)
├── postprocessing.py  # Visualization and post-analysis (Streamlit interface)
├── ml_leak_predict_from_onnx.py   # Machine Learning leak detection (ONNX model)
├── Leak_localisation_analysis.py  # Statistical and localization analysis
│
├── data/ # Raw and processed data
│ └── MAGNETO/
│     └── <sensor_ID>/ # Individual sensor folders
│         └── <date (YYYY-MM-DD)>/     # Test date
│             └── results/             # Output files (e.g., features.h5)
│             └── formatted_data/      # concatenated raw data (created during the after the preprocessing process)
│             └── labels/              # csv with annotations (created during the after the preprocessing process and hand made annotations based on formatted_data)
│
│
├── Wavely/ # Internal analysis modules
│ └── eda/ # Exploratory Data Analysis tools
│
├── plots/ # Generated figures and reports
│ └── <date (YYYY-MM-DD)>/ # Test date
│
│
├── <features_env/>
├── <leak_detection_env/>
│
└── README.md
```
