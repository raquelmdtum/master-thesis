# Multimodal Analysis of Environmental Influences on Emotion Using Mobile EEG

### Master Thesis by Raquel Marques (s243636), Spring 2026

This repository contains the code and analysis developed for my Master's thesis.

The project investigates how urban environments influence emotional and cognitive responses using multimodal data collected in the eMOTIONAL Cities project, including mobile EEG, eye-tracking, physiological signals, and environmental measurements.

## Setup 

Create the environment:

```
conda env create -f environment.yml
conda activate tese
```

## Data

The dataset is **not included** in this repository due to size and access restrictions.

All data are stored on the AIDRAGON server in the shared RAID storage:

```
/mnt/raid/emotional_data_raquel/
```

The code and notebooks in this repository access the dataset directly from this location.

## Structure

`notebooks/`  
Contains the main notebooks used throughout the thesis for dataset exploration, preprocessing, and analysis. The primary pipeline (`do_process.ipynb`) integrates the different data modalities (physiological signals, eye-tracking, environmental measurements, and EEG) into a unified dataset for subsequent analysis.

`utils/`   
Includes modular utility scripts that support the preprocessing pipeline. These modules implement functions for processing the different data modalities, handling participant metadata, and providing general helper functionality used across the project.

`schemas/`   
Defines the data schemas used to interpret the multimodal acquisition streams. These schemas specify how the sensor data are structured and synchronized when loading the recordings.

`helpers.py` / `modules.py`   
Provide shared helper functions and import wrappers used across notebooks to organize the codebase and simplify access to the utility modules.

## Status

Work in progress - thesis currently in development.
