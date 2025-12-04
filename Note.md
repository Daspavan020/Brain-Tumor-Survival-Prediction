🟢 Final Tip



Every time you open PowerShell, you must activate venv:

- cd D:\\FinalProject
  .\\venv\\Scripts\\Activate.ps1

-----------------------------------------------------------------------------------------------------------------------------------------------------------------


import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from lifelines.utils import concordance_index
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt

# ------------------------------------------------
# 1. Cox Partial Likelihood Loss (DeepSurv)
# ------------------------------------------------
class CoxLoss(tf.keras.losses.Loss):
    def call(self, y_true, y_pred):
        times = y_true[:, 0]
        events = y_true[:, 1]

        order = tf.argsort(times, direction='DESCENDING')
        times = tf.gather(times, order)
        events = tf.gather(events, order)
        scores = tf.gather(y_pred[:, 0], order)

        exp_scores = tf.exp(scores)
        risk_set = tf.cumsum(exp_scores)

        log_likelihood = scores - tf.math.log(risk_set + 1e-8)
        neg_log_likelihood = -tf.reduce_sum(log_likelihood * events) / (tf.reduce_sum(events) + 1e-8)

        return neg_log_likelihood


# ------------------------------------------------
# 2. Load Data
# ------------------------------------------------
df = pd.read_csv(r"D:\FinalProject\survival_data.csv")

y_time = df["time"].values
y_event = df["event"].values
X = df.drop(columns=["time", "event"])
X = pd.get_dummies(X)

feature_names = X.columns.tolist()

scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_test, t_train, t_test, e_train, e_test = train_test_split(
    X, y_time, y_event, test_size=0.2, random_state=42
)

y_train = np.vstack([t_train, e_train]).T
y_test = np.vstack([t_test, e_test]).T


# ------------------------------------------------
# 3. Build Model
# ------------------------------------------------
model = models.Sequential([
    layers.Input(shape=(X_train.shape[1],)),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(1)
])

model.compile(optimizer=tf.keras.optimizers.Adam(0.001),
              loss=CoxLoss())

model.summary()

# ------------------------------------------------
# 4. Train Model
# ------------------------------------------------
history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=30,
    batch_size=32,
    verbose=2
)


# ------------------------------------------------
# 5. C-index Calculation & Display Table
# ------------------------------------------------
def calculate_c_index(model, X, time, event):
    risk = model.predict(X).flatten()
    return concordance_index(time, -risk, event)

train_c_index = calculate_c_index(model, X_train, t_train, e_train)
test_c_index = calculate_c_index(model, X_test, t_test, e_test)

results_df = pd.DataFrame({
    "Dataset": ["Train", "Test"],
    "C-Index": [train_c_index, test_c_index]
})

print("\n===== MODEL PERFORMANCE (TABLE) =====\n")
print(results_df, "\n")


# ------------------------------------------------
# 6. Predict Risks for All Test Patients and Show Table
# ------------------------------------------------
test_pred = model.predict(X_test).flatten()
pred_df = pd.DataFrame({
    "Time": t_test,
    "Event": e_test,
    "Predicted Risk": test_pred
})

print("\n===== SAMPLE PREDICTIONS TABLE =====\n")
print(pred_df.head(), "\n")


# ------------------------------------------------
# 7. Plot Loss Curve
# ------------------------------------------------
plt.figure()
plt.plot(history.history['loss'], label="Train Loss")
plt.plot(history.history['val_loss'], label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training & Validation Loss")
plt.legend()
plt.show()


# ------------------------------------------------
# 8. Plot Predicted Risk Distribution
# ------------------------------------------------
plt.figure()
plt.hist(test_pred, bins=20)
plt.xlabel("Predicted Risk Score")
plt.ylabel("Frequency")
plt.title("Risk Score Distribution on Test Set")
plt.show()


# ------------------------------------------------
# 9. Predict for ONE Patient
# ------------------------------------------------
def predict_patient(patient_dict):
    df_input = pd.DataFrame([patient_dict])
    df_input = pd.get_dummies(df_input)

    for col in feature_names:
        if col not in df_input.columns:
            df_input[col] = 0

    df_input = df_input[feature_names]
    x_scaled = scaler.transform(df_input)

    risk_score = model.predict(x_scaled)[0][0]
    return risk_score


example_patient = {
    "Tumor Type": "Glioma",
    "Location": "Frontal Lobe",
    "Size (cm)": 5.4,
    "Grade": "III",
    "Patient Age": 45,
    "Gender": "Male"
}

print("\n===== ONE PATIENT PREDICTION =====")
print("Predicted Survival Risk =", predict_patient(example_patient))



-----------------------------------------------------------------------------------------------------------------------------------------------------------------






import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from lifelines.utils import concordance_index
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt

# ------------------------------------------------
# 1. Cox Partial Likelihood Loss (DeepSurv)
# ------------------------------------------------
class CoxLoss(tf.keras.losses.Loss):
    def call(self, y_true, y_pred):
        times = y_true[:, 0]
        events = y_true[:, 1]

        # Sort by descending time
        order = tf.argsort(times, direction='DESCENDING')
        times = tf.gather(times, order)
        events = tf.gather(events, order)
        scores = tf.gather(y_pred[:, 0], order)

        exp_scores = tf.exp(scores)
        risk_set = tf.cumsum(exp_scores)

        log_likelihood = scores - tf.math.log(risk_set + 1e-8)
        neg_log_likelihood = -tf.reduce_sum(log_likelihood * events) / (tf.reduce_sum(events) + 1e-8)

        return neg_log_likelihood


# ------------------------------------------------
# 2. Load Data
# ------------------------------------------------
df = pd.read_csv(r"D:\FinalProject\survival_data.csv")

y_time = df["time"].values
y_event = df["event"].values
X = df.drop(columns=["time", "event"])
X = pd.get_dummies(X)

feature_names = X.columns.tolist()

scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_test, t_train, t_test, e_train, e_test = train_test_split(
    X, y_time, y_event, test_size=0.2, random_state=42
)

y_train = np.vstack([t_train, e_train]).T
y_test = np.vstack([t_test, e_test]).T


# ------------------------------------------------
# 3. Build Model
# ------------------------------------------------
model = models.Sequential([
    layers.Input(shape=(X_train.shape[1],)),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(1)
])

model.compile(optimizer=tf.keras.optimizers.Adam(0.001),
              loss=CoxLoss())

model.summary()


# ------------------------------------------------
# 4. Train Model
# ------------------------------------------------
history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=30,
    batch_size=32,
    verbose=2
)


# ------------------------------------------------
# 5. C-index Calculation & Display Table
# ------------------------------------------------
def calculate_c_index(model, X, time, event):
    risk = model.predict(X).flatten()
    return concordance_index(time, -risk, event)   # negative risk: higher risk = shorter survival

train_c_index = calculate_c_index(model, X_train, t_train, e_train)
test_c_index = calculate_c_index(model, X_test, t_test, e_test)

results_df = pd.DataFrame({
    "Dataset": ["Train", "Test"],
    "C-Index": [train_c_index, test_c_index]
})

print("\n===== MODEL PERFORMANCE (TABLE) =====\n")
print(results_df, "\n")


# ------------------------------------------------
# 6. Predict risk for ALL patients in dataset
# ------------------------------------------------
all_risk_scores = model.predict(X).flatten()

# Start from original dataframe so all columns are included
pred_df = df.copy()

# Add patient ID (optional but useful)
pred_df.insert(0, "Patient_ID", range(1, len(df) + 1))

# Add predicted risk column
pred_df["Predicted_Risk"] = all_risk_scores

print("\n===== SAMPLE OF PATIENT RISK TABLE (FULL DATA) =====")
print(pred_df.head())



# ------------------------------------------------
# 7. Plot Loss Curve
# ------------------------------------------------
plt.figure()
plt.plot(history.history['loss'], label="Train Loss")
plt.plot(history.history['val_loss'], label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training & Validation Loss")
plt.legend()
plt.show()


# ------------------------------------------------
# 8. Plot Predicted Risk Distribution (on TEST SET)
# ------------------------------------------------
# compute test risk scores
test_pred = model.predict(X_test).flatten()

plt.figure()
plt.hist(test_pred, bins=20)
plt.xlabel("Predicted Risk Score")
plt.ylabel("Frequency")
plt.title("Risk Score Distribution on Test Set")
plt.show()


# ------------------------------------------------
# 9. Predict for ONE Patient
# ------------------------------------------------
def predict_patient(patient_dict):
    df_input = pd.DataFrame([patient_dict])
    df_input = pd.get_dummies(df_input)

    # align columns with training features
    for col in feature_names:
        if col not in df_input.columns:
            df_input[col] = 0

    df_input = df_input[feature_names]
    x_scaled = scaler.transform(df_input)

    risk_score = model.predict(x_scaled)[0][0]
    return risk_score


example_patient = {
    "Tumor Type": "Glioma",
    "Location": "Frontal Lobe",
    "Size (cm)": 5.4,
    "Grade": "III",
    "Patient Age": 45,
    "Gender": "Male"
}

print("\n===== ONE PATIENT PREDICTION =====")
print("Predicted Survival Risk =", predict_patient(example_patient))


# ------------------------------------------------
# 10. SAVE RESULTS TO ONE EXCEL FILE
# ------------------------------------------------
output_path = r"D:\FinalProject\survival_results.xlsx"  # change path if needed

# Make training history table
hist_df = pd.DataFrame({
    "Epoch": range(1, len(history.history['loss']) + 1),
    "Train Loss": history.history['loss'],
    "Validation Loss": history.history['val_loss']
})

with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
    # Save accuracy table
    results_df.to_excel(writer, sheet_name="Model Accuracy", index=False)

    # Save predictions table (ALL patients)
    pred_df.to_excel(writer, sheet_name="Predictions", index=False)

    # Save training & validation loss over epochs
    hist_df.to_excel(writer, sheet_name="Training History", index=False)

print(f"\n===== FILE SAVED =====\n{output_path}\n")
