import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from lifelines.utils import concordance_index
from lifelines import KaplanMeierFitter
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from datetime import datetime

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
#     + create Risk_Group + sort by risk
# ------------------------------------------------
all_risk_scores = model.predict(X).flatten()

# Start from original dataframe so all columns are included
pred_df = df.copy()

# Add patient ID
pred_df.insert(0, "Patient_ID", range(1, len(df) + 1))

# Add predicted risk column
pred_df["Predicted_Risk"] = all_risk_scores

# Define High / Low risk groups using median risk
median_risk = np.median(all_risk_scores)
pred_df["Risk_Group"] = np.where(pred_df["Predicted_Risk"] >= median_risk,
                                 "High", "Low")

# Sort by predicted risk (highest risk first)
pred_df = pred_df.sort_values("Predicted_Risk", ascending=False).reset_index(drop=True)

print("\n===== SAMPLE OF PATIENT RISK TABLE (FULL DATA) =====")
print(pred_df.head())


# ------------------------------------------------
# 7. Compute test predictions for plots
# ------------------------------------------------
test_pred = model.predict(X_test).flatten()


# ------------------------------------------------
# 8. Kaplan–Meier Curves for High vs Low Risk
# ------------------------------------------------
kmf = KaplanMeierFitter()


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
# 10. Paths with timestamp (Excel + PDF change every run)
# ------------------------------------------------
timestamp = datetime.now().strftime("%Y-%m-%d_%I-%M-%S_%p")
excel_path = fr"D:\FinalProject\survival_results_{timestamp}.xlsx"
pdf_path   = fr"D:\FinalProject\survival_report_{timestamp}.pdf"


# ------------------------------------------------
# 11. Build training history table (for Excel)
# ------------------------------------------------
hist_df = pd.DataFrame({
    "Epoch": range(1, len(history.history['loss']) + 1),
    "Train Loss": history.history['loss'],
    "Validation Loss": history.history['val_loss']
})


# ------------------------------------------------
# 12. SAVE RESULTS TO ONE EXCEL FILE
# ------------------------------------------------
with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
    # Accuracy (C-index)
    results_df.to_excel(writer, sheet_name="Model Accuracy", index=False)

    # Full patient data + risk + risk group (sorted)
    pred_df.to_excel(writer, sheet_name="Predictions", index=False)

    # Training history
    hist_df.to_excel(writer, sheet_name="Training History", index=False)

print(f"\n===== EXCEL FILE SAVED =====\n{excel_path}\n")


# ------------------------------------------------
# 13. GENERATE PDF REPORT (summary + curves)
# ------------------------------------------------
with PdfPages(pdf_path) as pdf:
    # --- Page 1: Text Summary ---
    fig, ax = plt.subplots(figsize=(8.27, 11.69))  # A4 size
    ax.axis('off')
    summary_text = (
        "SurvivalNet Report\n\n"
        f"Run timestamp: {timestamp}\n\n"
        f"Train C-index: {train_c_index:.4f}\n"
        f"Test C-index : {test_c_index:.4f}\n\n"
        f"Total patients: {len(df)}\n"
        f"High-risk patients: {(pred_df['Risk_Group']=='High').sum()}\n"
        f"Low-risk patients : {(pred_df['Risk_Group']=='Low').sum()}\n"
    )
    ax.text(0.5, 0.8, summary_text, ha='center', va='top', fontsize=14)
    pdf.savefig(fig)
    plt.close(fig)

    # --- Page 2: Loss Curve ---
    fig, ax = plt.subplots()
    ax.plot(history.history['loss'], label="Train Loss")
    ax.plot(history.history['val_loss'], label="Validation Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training & Validation Loss")
    ax.legend()
    ax.grid(True)
    pdf.savefig(fig)
    plt.close(fig)

    # --- Page 3: Risk Distribution (Test Set) ---
    fig, ax = plt.subplots()
    ax.hist(test_pred, bins=20)
    ax.set_xlabel("Predicted Risk Score")
    ax.set_ylabel("Frequency")
    ax.set_title("Risk Score Distribution (Test Set)")
    ax.grid(True)
    pdf.savefig(fig)
    plt.close(fig)

    # --- Page 4: Kaplan–Meier Curves for High vs Low Risk ---
    fig, ax = plt.subplots()
    for group, label in [("High", "High Risk"), ("Low", "Low Risk")]:
        mask = pred_df["Risk_Group"] == group
        kmf.fit(
            durations=pred_df.loc[mask, "time"],
            event_observed=pred_df.loc[mask, "event"],
            label=label
        )
        kmf.plot_survival_function(ax=ax)

    ax.set_title("Kaplan–Meier Survival Curves by Risk Group")
    ax.set_xlabel("Time")
    ax.set_ylabel("Survival Probability")
    ax.grid(True)
    pdf.savefig(fig)
    plt.close(fig)

print(f"\n===== PDF REPORT SAVED =====\n{pdf_path}\n")
