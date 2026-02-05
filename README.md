# Minor Project - LSTM-Based Energy Consumption Prediction

## 📌 Project Overview

This project implements a **Long Short-Term Memory (LSTM) neural network** to predict hourly electricity consumption using historical energy usage data.

The model learns temporal patterns from time-series data and forecasts future electricity demand. This helps in smart energy management, load forecasting, and efficient power utilization.


---


## 🎯 Objectives

- Perform preprocessing and exploratory data analysis (EDA) on energy consumption dataset
- Build and train an LSTM-based time-series forecasting model
- Predict future electricity consumption
- Evaluate model performance using regression metrics
- Visualize predictions and training performance
- Save trained model for reuse
  

---


## 📊 Dataset

The dataset contains **hourly building energy consumption data**.

### Features Include:
- Electricity usage
- Gas usage
- HVAC electricity consumption
- Cooling and heating load
- Lighting consumption
- Equipment electricity usage

### Target Variable:
Electricity:Facility kW


### Dataset Size:
- 8760 hourly samples (1 year)
  

---


## 🧠 Model Architecture

The model uses a **Univariate LSTM Neural Network**:

Input → LSTM Layer (50 units) → Dense Layer → Output

### Why LSTM?
LSTM networks are designed to:
- Capture long-term temporal dependencies
- Handle sequential data effectively
- Reduce vanishing gradient problems


---


## ⚙️ Project Workflow
→ Data Loading
→ Data Preprocessing
→ Exploratory Data Analysis
→ Train-Test Split
→ Data Scaling
→ Sequence Creation
→ LSTM Model Training
→ Prediction
→ Evaluation
→ Visualization
→ Model Saving


---


## 🧹 Data Preprocessing

- Fixed invalid timestamps (`24:00:00`)
- Converted timestamps to datetime format
- Sorted data chronologically
- Selected target electricity consumption feature


---


## 📈 Exploratory Data Analysis (EDA)

Generated the following plots:

- Full yearly electricity consumption trend
- Weekly consumption pattern
- Hourly usage behavior
- Distribution histogram of electricity consumption


---


## 🔄 Sequence Creation

The time-series data is converted into overlapping sequences:

- Uses last **24 hours** to predict next hour
- Generates 3D tensor input suitable for LSTM training


---


## 🏋️ Model Training

- Optimizer: Adam
- Loss Function: Mean Squared Error (MSE)
- Early stopping used to prevent overfitting


---


## 📏 Evaluation Metrics

The model performance is measured using:

- **RMSE (Root Mean Squared Error)**  
- **MAE (Mean Absolute Error)**  


---


## 📊 Output Visualizations

- Actual vs Predicted Electricity Consumption
- Training vs Validation Loss Graph
- Multiple EDA plots


---


## 💾 Saved Artifacts

| File | Description |
|--------|-------------|
| `electricity_lstm_model.h5` | Trained LSTM model |
| `y_pred.npy` | Model predictions |
| `y_test.npy` | Actual test values |
| `*.png` | Generated graphs |


---


## 📂 Project Structure
```
LSTMforEnergyPrediction/
│
├── main.py
├── requirements.txt
├── README.md
├── .gitignore
│
├── plots/
├── saved_models/
├── predictions/
```

---

📚 Technologies Used

Python

TensorFlow / Keras

NumPy

Pandas

Scikit-Learn

Matplotlib

---


🚀 Future Improvements

Multivariate LSTM using additional energy features

Multi-step future forecasting

Hyperparameter optimization

Real-time dashboard integration

Transformer-based time-series modeling

---

👨‍💻 Author

Sid (Siddhesh Khankhoje)
Computer Science Student | AI/ML Enthusiast

---


📜 License

This project is developed for academic and educational purposes.
