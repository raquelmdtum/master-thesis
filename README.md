# The Effect of Environmental Factors on Neuro-Physiological Responses in a Naturalistic Walk Experiment

### Master Thesis by Raquel Marques (s243636), Spring 2026

This repository contains the code and analysis developed for my Master's thesis, submitted to the Technical University of Denmark (DTU) as part of the MSc in Business Analytics. 

The project investigates how urban environments influence physiological and neural responses using multimodal data from the eMOTIONAL Cities Horizon 2020 project, collected across four Copenhagen neighbourhoods using mobile EEG, eye-tracking, ECG, EDA, GPS, and environmental sensors.

## Setup 

Create the environment:

```
conda env create -f environment.yml
conda activate tese
```

## Data

The dataset is **not included** in this repository due to size and access restrictions. All data are stored on the AIDRAGON server in the shared RAID storage:

```
/mnt/raid/emotional_data_raquel/
```

The code and notebooks in this repository access the dataset directly from this location. The computer vision pipeline (`run_panoptic_tracking.py`) was executed on the DTU HPC cluster, with data stored under `/work3/s243636/`.

## Structure

The repository is flat, meaning that all notebooks, scripts, and supporting files live directly in the `notebooks/` folder alongside this README, `environment.yml`, and `.gitignore`.

### Notebooks

**`do_process.ipynb`** — Loads raw recordings via the Pluma ecosystem and exports geodata, Empatica (EDA, HRV, HR), and EEG streams to CSV. Assembles the per-session `alldata.csv` files used by all downstream notebooks. Corresponds to the Data Processing chapter.

**`do_quality_testing.ipynb`** — Audits processed sessions for missing, empty, or corrupt files and produces signal-level quality diagnostics across modalities. Basis for participant/session exclusion decisions reported in the thesis.

**`do_typologies.ipynb`** — Assigns the custom urban typology labels (N, U, U+M, U+N, U+N+M) and a land-use class to each second of each session based on cumulative walking distance. Produces the typology and land-use maps.

**`do_eyetracking.ipynb`** — Preprocesses Pupil Labs gaze data and prepares frame-aligned fixation features. Feeds into `run_panoptic_tracking.py`. Methodology described in Section 4.1.

**`do_statistics.ipynb`** — Exploratory modelling notebook used during development. Contains GLM and LMM prototypes and spatial aggregation utilities. Informed the final modelling decisions but is not the source of reported results.

**`do_analysis.ipynb`** — Produces individual-level multimodal time series plots for qualitative inspection, anchored automatically to the peak EDA event per session.

**`do_results.ipynb`** — Primary analysis notebook. Runs all three analyses reported in Section 6: linear mixed-effects models, XGBoost with SHAP, and leave-one-participant-out (LOPO) cross-validation, across six physiological outcomes (HR, RMSSD, alpha, theta, tonic EDA, phasic EDA).

### Utilities & Schemas

**`utils/`** — Modular utility scripts supporting the preprocessing pipeline. Implements functions for processing each data modality (EDA, ECG, EEG, GPS, climate, eye-tracking) and general helper functionality used across notebooks.

**`schemas/`** — Data schemas defining how the multimodal acquisition streams are structured and synchronised when loading recordings via Pluma.

**`helpers.py`** — Shared helper functions used across notebooks, complementing `modules.py`.

### Scripts & Supporting Files

**`run_panoptic_tracking.py`** — HPC script. Processes eye-tracking video frame by frame using Mask2Former (panoptic segmentation) and YOLOv8+BoT-SORT (object tracking), computes optical flow for motion-compensated gaze classification, and outputs per-session gaze-environment fusion CSVs. Described in Section 4.1.

**`modules.py`** — Shared imports and Pluma initialisation wrappers used across the processing notebooks.

**`instances_val2017.json`** — COCO category label definitions (80 classes) used by the YOLOv8 model for object label mapping in `run_panoptic_tracking.py`.