import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping

import os

#train or load the ml model
TRAIN_MODEL = False   #kept true to train model, false to load model

#loadinf the dataset
df = pd.read_csv("USA_GA_Albany-Dougherty.County.AP.722160_TMY3_LOW.csv")

print("Initial shape:", df.shape)

#preprocessing
df["Date/Time"] = "2025 " + df["Date/Time"]

mask_24 = df["Date/Time"].str.contains("24:00:00")

df.loc[mask_24, "Date/Time"] = (
    df.loc[mask_24, "Date/Time"]
    .str.replace("24:00:00", "00:00:00", regex=False)
)

df["Date/Time"] = pd.to_datetime(
    df["Date/Time"], format="%Y %m/%d  %H:%M:%S"
)

df.loc[mask_24, "Date/Time"] += pd.Timedelta(days=1)

df = df.sort_values("Date/Time").set_index("Date/Time")


#target feature
target_col = "Electricity:Facility [kW](Hourly)"
series = df[target_col]


# EDA- analysis of dataset

# 1️⃣ Full Time Series
plt.figure(figsize=(12,4))
plt.plot(series)
plt.title("Full Electricity Consumption Time Series")
plt.xlabel("Time")
plt.ylabel("Electricity Consumption (kW)")
plt.savefig("eda_full_series.png")
plt.close()

# 2️⃣ First Week Pattern
plt.figure(figsize=(10,4))
plt.plot(series.iloc[:24*7])
plt.title("First Week Consumption Pattern")
plt.xlabel("Time (Hours)")
plt.ylabel("Electricity Consumption (kW)")
plt.savefig("eda_first_week.png")
plt.close()

# 3️⃣ First 48 Hours (Hourly Pattern)
plt.figure(figsize=(10,4))
plt.plot(series.iloc[:48])
plt.title("First 48 Hours Consumption Pattern")
plt.xlabel("Time (Hours)")
plt.ylabel("Electricity Consumption (kW)")
plt.savefig("eda_48_hours.png")
plt.close()

# 4️⃣ Distribution Histogram
plt.figure(figsize=(6,4))
plt.hist(series, bins=50)
plt.title("Electricity Consumption Distribution")
plt.xlabel("Time (Hours)")
plt.ylabel("Electricity Consumption (kW)")
plt.savefig("eda_distribution.png")
plt.close()


#train-test split
split_ratio = 0.8
split_index = int(len(series) * split_ratio)

train_series = series.iloc[:split_index]
test_series  = series.iloc[split_index:]


# scaling - to prevent domination of larger feature values(normalisation)
scaler = MinMaxScaler()

train_scaled = scaler.fit_transform(train_series.values.reshape(-1, 1))
test_scaled  = scaler.transform(test_series.values.reshape(-1, 1))


# sequence creation 
TIME_STEPS = 24

def create_sequences(data, time_steps):
    X, y = [], []
    for i in range(len(data) - time_steps):
        X.append(data[i:i + time_steps])
        y.append(data[i + time_steps])
    return np.array(X), np.array(y)

X_train, y_train = create_sequences(train_scaled, TIME_STEPS)
X_test, y_test   = create_sequences(test_scaled, TIME_STEPS)


# training / loading model
if TRAIN_MODEL or not os.path.exists("electricity_lstm_model.h5"):

    print("\nTraining Model...")

    model = Sequential([
        LSTM(50, activation="tanh", input_shape=(TIME_STEPS, 1)),
        Dense(1)
    ])

    model.compile(optimizer="adam", loss="mse")

    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True
    )

    history = model.fit(
        X_train,
        y_train,
        epochs=50,
        batch_size=32,
        validation_split=0.1,
        callbacks=[early_stop],
        verbose=1
    )

    model.save("electricity_lstm_model.h5")

else:
    print("\nLoading Saved Model...")
    model = load_model("electricity_lstm_model.h5", compile=False)

    model.compile(
    optimizer="adam",
    loss="mse"
    )
    history = None


# predicting
y_pred = model.predict(X_test)

y_test_inv = scaler.inverse_transform(y_test)
y_pred_inv = scaler.inverse_transform(y_pred)


# saving the predictions
np.save("y_test.npy", y_test_inv)
np.save("y_pred.npy", y_pred_inv)


# evaluating with rmse and mae
rmse = np.sqrt(mean_squared_error(y_test_inv, y_pred_inv))
mae  = mean_absolute_error(y_test_inv, y_pred_inv)

print("\nModel Evaluation:")
print("RMSE:", rmse)
print("MAE:", mae)


# graph plots
#1
plt.figure(figsize=(10,5))
plt.plot(y_test_inv, label="Actual")
plt.plot(y_pred_inv, label="Predicted")
plt.legend()
plt.title("Actual vs Predicted Electricity")
plt.savefig("actual_vs_predicted.png")
plt.close()


#2
if history is not None:
    plt.figure(figsize=(8,4))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.legend()
    plt.title("Training vs Validation Loss")
    plt.savefig("training_loss.png")
    plt.close()


print("\n✅ EDA plots, predictions, and model ready.")