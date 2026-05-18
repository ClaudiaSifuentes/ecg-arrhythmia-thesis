# ECG Arrhythmia Detection Project

This project focuses on the detection of cardiac arrhythmias using ECG signals and deep learning techniques. The primary dataset used is the MIT-BIH Arrhythmia Database. The architecture is designed to classify ECG beats, score anomalies over time, and detect sustained clinical events.

## Project Structure

```
ecg-arrhythmia-thesis
├── .vscode
│   ├── launch.json
│   ├── settings.json
│   └── tasks.json
├── configs
│   ├── default.yaml
│   ├── l1_beat_classifier.yaml
│   ├── l2_temporal_scoring.yaml
│   └── l3_event_detection.yaml
├── logs
│   └── .gitkeep
├── notebooks
│   └── README.md
├── scripts
│   ├── download_mitbih.py
│   ├── preprocess_ecg.py
│   ├── train_l1.py
│   ├── score_l2.py
│   └── detect_l3.py
├── src
│   ├── __init__.py
│   ├── cli.py
│   ├── config
│   │   ├── __init__.py
│   │   ├── loader.py
│   │   └── schema.py
│   ├── data
│   │   ├── __init__.py
│   │   ├── mitbih_dataset.py
│   │   ├── preprocessing.py
│   │   ├── rr_features.py
│   │   └── splits.py
│   ├── inference
│   │   ├── __init__.py
│   │   ├── l1_predict.py
│   │   ├── l2_score.py
│   │   └── l3_events.py
│   ├── models
│   │   ├── __init__.py
│   │   ├── cnn1d.py
│   │   └── fusion.py
│   ├── training
│   │   ├── __init__.py
│   │   ├── losses.py
│   │   ├── metrics.py
│   │   ├── trainer.py
│   │   └── callbacks.py
│   ├── utils
│   │   ├── __init__.py
│   │   ├── logging.py
│   │   ├── paths.py
│   │   ├── reproducibility.py
│   │   └── timers.py
│   └── version.py
├── tests
│   ├── __init__.py
│   ├── test_rr_features.py
│   ├── test_splits_patientwise.py
│   └── test_l2_l3_logic.py
├── .env.example
├── .gitignore
├── environment.yml
├── requirements.txt
├── pyproject.toml
├── README.md
└── run.py
```

## Installation

To set up the project, you can create a Conda environment using the provided `environment.yml` file:

```bash
conda env create -f environment.yml
conda activate ecg-arrhythmia-thesis
```

Alternatively, you can install the required packages using pip:

```bash
pip install -r requirements.txt
```

## Usage

1. **Download the MIT-BIH dataset**: Run the script to download the dataset.
   ```bash
   python scripts/download_mitbih.py
   ```

2. **Preprocess the ECG data**: Preprocess the downloaded ECG signals.
   ```bash
   python scripts/preprocess_ecg.py
   ```

3. **Train the L1 beat classification model**:
   ```bash
   python scripts/train_l1.py
   ```

4. **Compute L2 anomaly scores**:
   ```bash
   python scripts/score_l2.py
   ```

5. **Detect clinical events (L3)**:
   ```bash
   python scripts/detect_l3.py
   ```

## Configuration

Configuration files are located in the `configs` directory. The `default.yaml` file contains centralized configuration parameters for the project. Specific configurations for each level of classification can be found in their respective YAML files.

## Logging

Logging is configured to capture important events and errors during execution. Logs are stored in the `logs` directory.

## Testing

Unit tests are provided in the `tests` directory. You can run the tests using:

```bash
pytest tests/
```

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.

## License

This project is licensed under the MIT License. See the LICENSE file for more details.

## Acknowledgments

- MIT-BIH Arrhythmia Database for providing the dataset.
- PyTorch for the deep learning framework.