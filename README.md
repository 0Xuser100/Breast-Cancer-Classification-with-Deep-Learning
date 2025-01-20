# Breast Cancer Classification with Deep Learning

This project demonstrates an end-to-end deep learning workflow for classifying breast cancer tumors as **malignant** or **benign** using the Breast Cancer dataset from `sklearn.datasets`. The project includes data preprocessing, model building, training, evaluation, and saving the trained model.

---

## Table of Contents
1. [Project Overview](#project-overview)
2. [Dataset](#dataset)
3. [Requirements](#requirements)
4. [Setup](#setup)
5. [Code Structure](#code-structure)
6. [Results](#results)
7. [License](#license)
8. [Acknowledgments](#acknowledgments)

---

## Project Overview
The goal of this project is to build a deep learning model to classify breast cancer tumors based on features such as radius, texture, and perimeter. The model is developed using TensorFlow and Keras, and the dataset is preprocessed using `scikit-learn`.

---

## Dataset
The dataset used in this project is the **Breast Cancer Wisconsin (Diagnostic) Dataset**, available in `sklearn.datasets`. It contains 569 samples with 30 features each, and a target variable indicating whether the tumor is malignant (0) or benign (1).

### Dataset Features
- **Features**: 30 numeric features (e.g., mean radius, mean texture, mean perimeter, etc.).
- **Target**: Binary classification (0 = malignant, 1 = benign).

### Dataset Source
The dataset can be accessed using the following code snippet:
```python
from sklearn.datasets import load_breast_cancer
data = load_breast_cancer()
```

---

## Requirements
To run this project, you need the following Python libraries:
- `numpy`
- `pandas`
- `scikit-learn`
- `tensorflow`
- `matplotlib`

Install dependencies using the `requirements.txt` file:
```bash
pip install -r requirements.txt
```

---

## Setup
1. Clone this repository:
   ```bash
   git clone https://github.com/your-username/breast-cancer-classification.git
   cd breast-cancer-classification
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the script:
   ```bash
   python breast_cancer_dl.py
   ```

---

## Code Structure
The project consists of the following steps:
1. **Data Loading**: Load the Breast Cancer dataset using `sklearn.datasets`.
2. **Data Preprocessing**: Split the dataset into training and testing sets and standardize the features.
3. **Model Building**: Construct a deep learning model using TensorFlow and Keras.
4. **Model Training**: Train the model on the training dataset.
5. **Model Evaluation**: Evaluate the model on the test dataset.
6. **Model Saving**: Save the trained model as `breast_cancer_model.h5`.

---

## Results
- The model achieves an accuracy of **98.2%** on the test set.
- The trained model is saved as `breast_cancer_model.h5`.

---

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## Acknowledgments
- Dataset: [Breast Cancer Wisconsin (Diagnostic) Dataset](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_breast_cancer.html)
- Libraries used: `scikit-learn`, `TensorFlow`, `pandas`, `matplotlib`

