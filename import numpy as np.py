import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from lifelines.utils import concordance_index
import tensorflow as tf
from tensorflow.keras import layers, models

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
# 2. Load Survival Dataset
# ------------------------------------------------
df = pd.read_csv(r"D:\Final Project\survival_data.csv")

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
# 3. Build SurvivalNet Model (DeepSurv)
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
# 4. Train the Model
# ------------------------------------------------
model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=30,
    batch_size=32,
    verbose=2
)


# ------------------------------------------------
# 5. C-index Calculation (ACCURACY)
# ------------------------------------------------
def calculate_c_index(model, X, time, event):
    risk = model.predict(X).flatten()
    return concordance_index(time, -risk, event)   # negative risk because higher risk = shorter survival


train_c_index = calculate_c_index(model, X_train, t_train, e_train)
test_c_index = calculate_c_index(model, X_test, t_test, e_test)

print("\n==========================================")
print(" MODEL ACCURACY (C-INDEX) ")
print("==========================================")
print(f"Train C-index : {train_c_index:.4f}")
print(f"Test C-index  : {test_c_index:.4f}")
print("==========================================\n")


# ------------------------------------------------
# 6. Predict survival risk for ONE patient
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


# Example Patient
example_patient = {
    "Tumor Type": "Glioma",
    "Location": "Frontal Lobe",
    "Size (cm)": 5.4,
    "Grade": "III",
    "Patient Age": 45,
    "Gender": "Male"
}

print("Predicted Survival Risk =", predict_patient(example_patient))