# 🎓 Students Marks Prediction Using Neural Networks

An AI-powered system that predicts student final marks based on their quiz performance patterns using deep learning. Built with MongoDB, TensorFlow/Keras, and Python.
## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [System Architecture](#system-architecture)
- [Installation](#installation)
- [Usage](#usage)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## 🎯 Overview

This project implements a deep learning solution to predict student final marks in an e-learning environment. The system analyzes quiz performance patterns and generates predictions using an artificial neural network (ANN).

**Key Capabilities:**
- Predicts final marks based on quiz scores
- Identifies at-risk learners early
- Provides insights for personalized learning interventions
- Handles 2,500+ students efficiently

## ✨ Features

- **Automated Data Generation**: Synthetic educational data creation with realistic patterns
- **Comprehensive Preprocessing**: Handles missing values, duplicates, and outliers
- **Feature Engineering**: Extracts meaningful patterns from raw quiz data
- **Deep Neural Network**: 4-layer architecture with dropout regularization
- **Model Persistence**: Saves trained model for future predictions
- **Performance Metrics**: Evaluates using MAE and MSE

## 🏗️ System Architecture

```
MongoDB Database
    ↓
Data Loading & Preprocessing
    ↓
Feature Engineering (5 features)
    ↓
Neural Network Training
    ↓
Model Evaluation & Saving
```

### Database Collections

- **learners**: Student records (2,500 students)
- **teachers**: Instructor data (50 teachers)
- **specialties**: Academic programs (4 specialties)
- **subjects**: Course subjects (4 subjects)
- **quizzes**: Assessment items (40 quizzes)
- **answers**: Quiz responses with scores
- **marks**: Final marks by student and subject

## 🔧 Installation

### Prerequisites

- Python 3.8 or higher
- MongoDB 4.x or higher
- pip package manager

### Step 1: Clone the Repository

```bash
git clone https://github.com/yourusername/students-marks-prediction.git
cd students-marks-prediction
```

### Step 2: Install Dependencies

```bash
pip install pymongo pandas numpy scikit-learn tensorflow keras faker matplotlib seaborn
```

Or use requirements.txt:

```bash
pip install -r requirements.txt
```

### Step 3: Start MongoDB

```bash
# On Linux/Mac
sudo systemctl start mongod

# On Windows
net start MongoDB
```

## 🚀 Usage

### Step 1: Generate Synthetic Data

Run the data generation script to populate the MongoDB database:

```bash
python generate_data.py
```

**Output:**
```
✅ Database filled with realistic data:
   • 2500 learners
   • 50 teachers
   • 40 quizzes
   • ~25000 answers
   • 10000 marks
```

### Step 2: Train the Model

Execute the training script to build and train the neural network:

```bash
python Ann_trainmodel.py
```

**Expected Output:**
```
🧹 Starting data preprocessing...
✅ Preprocessing complete
✅ Data ready: 2500 learners, 5 features

Epoch 1/100
...
📊 Test Mean Absolute Error: 2.45
📉 Test Mean Squared Error: 8.32

💾 Model saved as 'student_performance_model.h5'
```

### Step 3: Make Predictions

Load the saved model and make predictions on new data:

```python
from keras.models import load_model
import numpy as np

# Load trained model
model = load_model('student_performance_model.h5')

# Prepare new student data (5 features)
new_student = np.array([[15.5, 18.0, 12.0, 2.3, 10]])

# Scale features (use same scaler from training)
new_student_scaled = scaler.transform(new_student)

# Predict mark
predicted_mark = model.predict(new_student_scaled)
print(f"Predicted Mark: {predicted_mark[0][0]:.2f}")
```

## 📊 Dataset

### Synthetic Data Generation

The system generates realistic educational data using the Faker library:

- **2,500 learners** across 4 specialties
- **50 teachers** with specialty assignments
- **40 quizzes** (10 per subject)
- **Variable engagement**: Each learner takes 5-15 quizzes
- **Realistic scoring**: Correct rates from 30%-100%

### Features

| Feature | Description |
|---------|-------------|
| `mean_score` | Average quiz score |
| `max_score` | Best quiz score |
| `min_score` | Worst quiz score |
| `std_score` | Score standard deviation (consistency) |
| `quiz_count` | Number of quizzes taken |

## 🧠 Model Architecture

### Neural Network Structure

```
Input Layer:    5 features
    ↓
Hidden Layer 1: 64 neurons + ReLU + Dropout(0.2)
    ↓
Hidden Layer 2: 32 neurons + ReLU + Dropout(0.2)
    ↓
Hidden Layer 3: 16 neurons + ReLU
    ↓
Output Layer:   1 neuron (Linear activation)
```

### Hyperparameters

- **Optimizer**: Adam (learning_rate=0.001)
- **Loss Function**: Mean Squared Error (MSE)
- **Epochs**: 100
- **Batch Size**: 32
- **Validation Split**: 10%

## 📈 Results

### Model Performance

- **Test MAE**: ~2.45 (average error of ±2.45 marks)
- **Test MSE**: ~8.32
- **Training Time**: ~2-3 minutes on CPU

### Sample Predictions

```
Predicted: 15.23 | Actual: 15.80
Predicted: 12.45 | Actual: 11.90
Predicted: 18.67 | Actual: 18.20
```

## 📁 Project Structure

```
students-marks-prediction/
│
├── generate_data.py          # Synthetic data generation
├── Ann_trainmodel.py          # Model training script
├── requirements.txt           # Python dependencies
├── README.md                  # Project documentation
│
├── models/
│   └── student_performance_model.h5  # Saved trained model
│
└── docs/
    └── technical_report.md    # Detailed technical report
```

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- TensorFlow/Keras for deep learning framework
- MongoDB for flexible NoSQL database
- Scikit-learn for preprocessing utilities
- Faker for synthetic data generation


⭐ If you find this project useful, please consider giving it a star!

## 🔮 Future Enhancements

- [ ] Add temporal features (learning progression over time)
- [ ] Implement cross-validation
- [ ] Create web dashboard for predictions
- [ ] Add subject-specific prediction models
- [ ] Integrate real-time prediction API
- [ ] Implement SHAP values for model explainability
