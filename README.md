# COVID-19 Patient Intubation Prediction

This repository contains the code the project **COVID-19 Patient Intubation Prediction**. The project aims to predict the likelihood of COVID-19 patients requiring intubation using a variety of machine learning classification techniques, including logistic regression, k-nearest neighbors, and artificial neural network, all implemented from scratch.

## Project Overview

COVID-19 can lead to severe respiratory complications, necessitating intubation in critical cases. Early prediction of this need can aid in timely medical interventions, optimizing resource allocation and improving patient outcomes. This project leverages clinical data to develop a predictive model for intubation in COVID-19 patients.
<p> <img src="https://github.com/user-attachments/assets/34c65fc0-8083-4107-8908-bef3da8752ad" width="1000"> </p> 

## Repository Contents

- **[ENGR_518_Project_final.ipynb](ENGR_518_Project_final.ipynb)**: Jupyter Notebook containing the project code and analysis.

## Dataset

The dataset used for this project was sourced from Kaggle: [COVID-19 Dataset](https://www.kaggle.com/datasets/meirnizri/covid19-dataset).

- **Description**: The dataset contains clinical features and outcomes for COVID-19 patients, including whether they required intubation.
  <p> <img src="https://github.com/user-attachments/assets/167fca23-dce5-4e1f-b65f-bdb3b0d9acc3" width="1000"> </p> 

- **Data Preprocessing**: Missing values were handled, and categorical features were encoded. Feature selection was performed to retain only the most relevant predictors.

## Features

The project followed these key steps:

1. **Data Collection and Preprocessing**: 
Clinical data of COVID-19 patients was sourced and cleaned for model training.
  
2. **Feature Engineering**: 
Relevant features were extracted and engineered to improve model performance.
  
3. **Model Development**: 
Various machine learning models were explored and evaluated to predict intubation likelihood, including the following:
    
    a) Logistic Regression
    <p> <img src="https://github.com/user-attachments/assets/84911a95-9056-4045-9f7c-e9bdb0b18956" width="1000"> </p> 

    b) K-Nearest Neighbors (KNN)
    <p> <img src="https://github.com/user-attachments/assets/20c65639-a7dd-466a-908e-2cd3ec340198" width="1000"> </p> 

    c) Artificial Neural Network (ANN)
    <p> <img src="https://github.com/user-attachments/assets/78958246-a214-49f1-a5f1-d8431f815a0f" width="1000"> </p> 

5. **Evaluation and Results**: 
    - The models were assessed based on accuracy, precision, recall, and other performance metrics.

## Requirements

To run the code in the Jupyter Notebook, you will need the following Python libraries:

- `pandas`
- `numpy`
- `scikit-learn`
- `matplotlib`
- `seaborn`

## Usage

1. Clone this repository:
   ```bash
   git clone https://github.com/AminLari/COVID-19-Patient-Intubation-Prediction.git

2. Install the required Python packages:
   ```bash
   pip install pandas numpy scikit-learn matplotlib seaborn

3. Open the Jupyter Notebook and run the cells:
   ```bash
   jupyter notebook ENGR_518_Project_final.ipynb

## Results

The final model achieved promising results, indicating its potential to assist healthcare providers in predicting the need for patient intubation.
### Testing Accuracy

| Classification Algorithm                       | Accuracy (%) - Original Dataset | Accuracy (%) - Reduced Dataset with PCA |
|------------------------------------------------|----------------------------------|-----------------------------------------|
| Built-in Logistic Regression (Cross-Entropy)  | 97.004                           | 91.92                                   |
| Logistic Regression with MSE                  | 92                               | 86.15                                   |
| Logistic Regression with Binary Cross-Entropy | **96.85**                            | 91.61                                   |
| Built-in KNN                                  | 96.08                            | 95.45                                   |
| KNN                                           | 95.98                            | **92.63**                                   |
| Built-in Neural Network                       | 96.63                            | 93.15                                   |
| Fully Connected Neural Network                            | 96.63                            | 86.58                                   |

## 📞 Contact
If you have any questions, feedback, or suggestions regarding this project, feel free to reach out:

- **Name**: Mohammadamin Lari  
- **Email**: [mohammadamin.lari@gmail.com](mailto:mohammadamin.lari@gmail.com)  
- **GitHub**: [AminLari](https://github.com/aminlari)

You are welcome to create issues or pull requests to improve the project. Contributions are highly appreciated! 
