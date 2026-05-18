# README for Notebooks

# ECG Arrhythmia Detection Notebooks

This directory contains Jupyter notebooks used for exploratory data analysis, model training, evaluation, and visualization related to the ECG arrhythmia detection project.

## Contents

- **Exploratory Data Analysis (EDA)**: Notebooks for visualizing and understanding the MIT-BIH Arrhythmia Database, including signal characteristics and feature distributions.
  
- **Model Training**: Notebooks that demonstrate the training process for the L1 beat classification model, including hyperparameter tuning and performance evaluation.

- **Anomaly Scoring**: Notebooks that illustrate the L2 temporal scoring method, showcasing how to compute anomaly scores based on the trained model.

- **Event Detection**: Notebooks that detail the L3 event detection logic, including the implementation of clinical event detection based on the anomaly scores.

## Usage

To run the notebooks, ensure you have the required dependencies installed. You can set up your environment using the provided `requirements.txt` or `environment.yml` files.

1. Install dependencies:
   - Using pip: `pip install -r requirements.txt`
   - Using conda: `conda env create -f environment.yml`

2. Launch Jupyter Notebook:
   ```bash
   jupyter notebook
   ```

3. Open the desired notebook and follow the instructions within.

## Notes

- Ensure that the MIT-BIH dataset is downloaded and accessible as specified in the project structure.
- The notebooks are designed to be modular and can be run independently or in sequence, depending on your analysis needs.

For any questions or issues, please refer to the main project documentation or contact the project maintainers.