import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Load saved data
y_true = np.load("lstm_test.npy")
lstm_pred = np.load("lstm_pred.npy")
trans_pred = np.load("transformer_pred.npy")

#evaluation metrics
def evaluate(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-10)))
    return rmse, mae, mape

lstm_rmse, lstm_mae, lstm_mape = evaluate(y_true, lstm_pred)
trans_rmse, trans_mae, trans_mape = evaluate(y_true, trans_pred)

print("\n📊 MODEL COMPARISON\n")

print("LSTM:")
print(f"RMSE: {lstm_rmse:.4f}, MAE: {lstm_mae:.4f}, MAPE: {lstm_mape:.4f}")

print("\nTransformer:")
print(f"RMSE: {trans_rmse:.4f}, MAE: {trans_mae:.4f}, MAPE: {trans_mape:.4f}")

#graph plots
plt.figure(figsize=(12,6))

plt.plot(y_true[:500], label="Actual", linewidth=2)
plt.plot(lstm_pred[:500], label="LSTM")
plt.plot(trans_pred[:500], label="Transformer")

plt.legend()
plt.title("LSTM vs Transformer Comparison")
plt.xlabel("Time")
plt.ylabel("Electricity (kW)")

plt.savefig("model_comparison.png")
plt.show()