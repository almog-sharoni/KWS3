# KWS3 Project

## Overview
The KWS3 project is designed for keyword spotting using deep learning techniques. It includes various components such as data loading, model definition, training, evaluation, and ablation studies to analyze the impact of different configurations on model performance.

## Directory Structure
```
KWS3
├── src
│   ├── data
│   │   ├── __init__.py
│   │   ├── config.py
│   │   └── data_loader.py
│   ├── models
│   │   ├── __init__.py
│   │   └── model.py
│   └── utils
│       ├── __init__.py
│       ├── augmentations.py
│       ├── train_utils.py
│       └── utils.py
├── experiments
│   └── ablation_results
│       └── README.md
├── notebooks
│   └── analysis.ipynb
├── ablation_study.py
├── train.py
├── evaluate.py
├── requirements.txt
└── README.md
```

## Installation
To set up the project, clone the repository and install the required dependencies:

```bash
pip install -r requirements.txt
```

## Usage
1. **Data Preparation**: Ensure that the datasets are available and properly configured in `src/data/config.py`.
2. **Training the Model**: Run the training script:
   ```bash
   python train.py
   ```
3. **Evaluating the Model**: After training, evaluate the model using:
   ```bash
   python evaluate.py
   ```
4. **Ablation Studies**: To run ablation tests, execute:
   ```bash
   python ablation_study.py --epochs <number_of_epochs>
   ```

## Components
- **Data Module**: Handles loading and preprocessing of datasets.
- **Model Module**: Contains the architecture for the keyword spotting model.
- **Utils Module**: Provides utility functions for training, logging, and data augmentation.
- **Experiments**: Contains results from ablation studies and analysis notebooks.

## Contributing
Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.

## License
This project is licensed under the MIT License. See the LICENSE file for details.