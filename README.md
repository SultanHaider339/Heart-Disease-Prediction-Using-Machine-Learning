# Heart Disease Prediction with Feature Selection and Neural Network

## Overview
This code implements a machine learning pipeline for predicting heart disease using:
1. Hybrid feature selection (Logistic Regression + Gradient Boosting)
2. A deep neural network classifier

## Feature Selection Process
1. **Two Selection Methods**:
   - `SelectFromModel` with Logistic Regression (max_iter=1000)
   - `SelectFromModel` with Gradient Boosting (100 estimators)

2. **Hybrid Selection**:
   - Combines features selected by either method using `np.union1d`
   - Final selected features: `['ca', 'chol', 'cp', 'exang', 'oldpeak', 'restecg', 'sex', 'thal']`

## Neural Network Architecture
A 5-layer sequential model:
1. **Input Layer**: 64 neurons (ReLU activation)
2. **Hidden Layers**:
   - 32 neurons (ReLU)
   - 16 neurons (ReLU)
   - 8 neurons (ReLU)
3. **Output Layer**: 1 neuron (Sigmoid activation for binary classification)

## Training Configuration
- **Optimizer**: Adam
- **Loss Function**: Binary crossentropy
- **Metrics**: Accuracy
- **Training Parameters**:
  - 50 epochs
  - Batch size of 32

## Performance
The model achieved **83.52% accuracy** on the test set.

## Key Techniques
- Ensemble feature selection combining linear and tree-based methods
- Deep learning approach with multiple hidden layers
- Binary classification for heart disease prediction
