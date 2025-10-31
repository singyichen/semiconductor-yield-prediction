# 🏭 Semiconductor Yield Prediction System

A machine learning system for predicting semiconductor manufacturing yield, detecting defective products early, and optimizing production efficiency.

## 📊 Project Overview

- **Dataset**: SECOM Dataset (1567 samples, 590 features)

- **Objective**: Predict whether a product is defective (binary classification)

- **Challenges**: High dimensionality, many missing values, severe class imbalance (only 6.5% defective samples)

- **Best Model**: XGBoost (F1 = 0.78, AUC = 0.92)

## 🚀 Quick Start

### Install Dependencies
```bash
pip install -r requirements.txt
```

### Train the Model
```bash
python src/model_training.py
```

### Launch API Service
```bash
cd api
python app.py
```


The API will be available at http://localhost:8000

### Deploy with Docker
```bash
docker build -t yield-predictor .
docker run -p 8000:8000 yield-predictor
```

## 📁 Project Structure
```
semiconductor-yield-prediction/
├── data/               # Raw and processed data
├── notebooks/          # Jupyter notebooks (EDA, modeling)
├── src/                # Core source code
│   ├── data_preprocessing.py
│   ├── feature_engineering.py
│   └── model_training.py
├── api/                # FastAPI service
├── models/             # Trained model files
├── outputs/            # Figures and reports
└── tests/              # Unit tests
```

## 🔧 Core Features

### 1. Data Preprocessing

- Missing value imputation (median fill)

- Outlier detection (Isolation Forest)

- Feature standardization

### 2. Feature Engineering

- Remove highly correlated features (reduce multicollinearity)

- Feature selection via Random Forest importance

- Generate interaction features

### 3. Model Training

- Handle class imbalance (SMOTE + under-sampling)

- Compare multiple models (RF, XGBoost, LightGBM)

- Hyperparameter optimization

- Cross-validation

### ⚙️ API Examples
### Single Prediction

**Request::**
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "features": {
      "sensor_0": 1.23,
      "sensor_1": -0.45,
      "sensor_2": 2.67,
      "sensor_3": 0.89,
      "sensor_4": -1.12,
      "sensor_5": 0.34,
      "sensor_6": 1.56,
      "sensor_7": -0.78,
      "sensor_8": 2.01,
      "sensor_9": 0.45
    }
  }'
```

**Response:**
```json
{
  "prediction": "Defect",
  "defect_probability": 0.8532,
  "confidence": 0.8532,
  "timestamp": "2024-10-31T10:30:00.123456",
  "alert": true
}
```

### Batch Prediction

**Request:**
```bash
curl -X POST "http://localhost:8000/predict/batch" \
  -H "Content-Type: application/json" \
  -d '{
    "batch": [
      {
        "sensor_0": 1.23,
        "sensor_1": -0.45,
        "sensor_2": 2.67,
        "sensor_3": 0.89,
        "sensor_4": -1.12,
        "sensor_5": 0.34,
        "sensor_6": 1.56,
        "sensor_7": -0.78,
        "sensor_8": 2.01,
        "sensor_9": 0.45
      },
      {
        "sensor_0": -0.56,
        "sensor_1": 1.23,
        "sensor_2": -1.45,
        "sensor_3": 0.67,
        "sensor_4": 2.34,
        "sensor_5": -0.89,
        "sensor_6": 1.12,
        "sensor_7": 0.45,
        "sensor_8": -1.67,
        "sensor_9": 0.78
      }
    ]
  }'
```

**Response:**
```json
{
  "predictions": [
    {
      "prediction": "Defect",
      "defect_probability": 0.8532,
      "confidence": 0.8532,
      "timestamp": "2024-10-31T10:30:00.123456",
      "alert": true
    },
    {
      "prediction": "Good",
      "defect_probability": 0.2341,
      "confidence": 0.7659,
      "timestamp": "2024-10-31T10:30:00.234567",
      "alert": false
    }
  ],
  "total": 2,
  "defect_count": 1,
  "defect_rate": 0.5
}
```

## 📈 Model Performance
| Model | ROC-AUC | F1 Score | Precision | Recall |
|-------|---------|----------|-----------|--------|
| Random Forest | 0.89 | 0.72 | 0.71 | 0.73 |
| **XGBoost** | **0.92** | **0.78** | **0.76** | **0.80** |
| LightGBM | 0.90 | 0.75 | 0.73 | 0.77 |

## 🎯 Business Value

1. **Early Defect Detection**: Predict defective products in real time, reducing defect outflow by up to 30%.

2. **Cost Reduction**: Minimize rework and material waste.

3. **Process Optimization**: Identify key factors affecting yield and improve production parameters.

4. **Decision Support**: Enable data-driven process adjustments and insights.

## 📊 Key Insights

### Top 5 sensors impacting yield:

1. Temperature Control (sensor_27)

2. Pressure Stability (sensor_45)

3. Etching Time (sensor_12)

4. Chemical Concentration (sensor_33)

5. Equipment Vibration (sensor_58)

🔄 Future Improvements

 Integrate LSTM for temporal dependency modeling

 Apply SHAP for explainable AI insights

 Integrate with MES systems

 Real-time monitoring dashboard (Grafana)

 Model drift detection & automatic retraining

## 👩‍💻 Author

[Mandy] – Data Scientist

GitHub: [singyichen](https://github.com/singyichen)

Email: ms.mandy610425@gmail.com

## 📝 License

MIT License