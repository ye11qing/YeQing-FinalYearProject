
# Time Series Forecasting for Traffic Prediction

## Introduction
This project focuses on the application of machine learning techniques for time series forecasting in the context of traffic prediction, particularly addressing challenges posed by missing data and optimizing the model structure.

## Proposed Approach
The approach combines mean imputation for handling missing values with advanced RNN models such as Gated Recurrent Units (GRU) and T-GCN models. This involves:

- **Mean Imputation**: Missing values are replaced with the average of each feature within the existing data.
- **T-GRU Model**: Enhances prediction accuracy by incorporating spatial dependencies between traffic network nodes through graph convolutions.
- **Hyperparameter Optimization**: Utilizes the Optuna library to fine-tune model parameters.

## Experiments
Three main experiments are conducted:
1. **GRU vs. T-GRU Performance**: Evaluates the effectiveness of spatial dependencies.
2. **Long-term Prediction**: Uses a sliced data processing approach and an encoder-decoder structure similar to transformers.
3. **Model Generalization**: Tests the model on a different dataset to explore transferability.

## Web Application
Developed using Flask and Flask-RESTX, the application provides a user-friendly interface for:
- Time series forecasting
- Model training
- Result visualization

## Conclusion
The thesis presents a comprehensive framework for effective traffic prediction, contributing to improved traffic management and urban planning.

## How to Run the Code

### Requirements
- Python 3.8+
- Libraries: numpy, pandas, torch, flask, flask-restx, optuna

### Setup
```bash
pip install numpy pandas torch flask flask-restx optuna
```

### Running the Application
```bash
python app.py
```

This will start the Flask server and make the web application accessible locally for time series forecasting and visualization tasks.