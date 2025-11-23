import gradio as gr
import pandas as pd
import joblib
import os
import warnings

# --- Silence sklearn's "feature names" warning when using numpy arrays ---
warnings.filterwarnings(
    "ignore",
    message="X does not have valid feature names, but",
    category=UserWarning,
)

# --- Paths to saved model + scaler (adjust names if yours differ) ---
MODEL_PATH = os.path.join("models", "fraud_detection_model.joblib")
SCALER_PATH = os.path.join("models", "scaler.joblib")

# Load model and scaler once at startup
model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

# SAME FEATURE ORDER AS DURING TRAINING:
# Time, V1..V28, Amount
REQUIRED_FEATURES = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]


def predict_from_csv(file):
    """
    Gradio handler:
    - Takes an uploaded CSV file
    - Loads it into a DataFrame
    - Applies the same preprocessing (scaling Time & Amount)
    - Runs the fraud detection model
    - Returns the DataFrame with prediction columns added
    """
    if file is None:
        return pd.DataFrame(columns=["Error"]), "Please upload a CSV file."

    # 1) Read CSV
    try:
        df = pd.read_csv(file.name)
    except Exception as e:
        return pd.DataFrame(columns=["Error"]), f"Error reading CSV: {e}"

    original_rows = len(df)

    # 2) Optional: limit rows for faster prediction (good for UI + HuggingFace)
    max_rows = 10000
    if original_rows > max_rows:
        df = df.head(max_rows)

    # 3) Drop Class column if user uploads original Kaggle data
    if "Class" in df.columns:
        df_features = df.drop("Class", axis=1).copy()
    else:
        df_features = df.copy()

    # 4) Check that all required columns are present
    missing_cols = [col for col in REQUIRED_FEATURES if col not in df_features.columns]
    if missing_cols:
        msg = (
            f"Missing required columns: {missing_cols}. "
            f"Expected at least these columns: {REQUIRED_FEATURES}"
        )
        return pd.DataFrame(columns=["Error"]), msg

    # 5) Keep only the required features (ignore extra columns if any)
    X = df_features[REQUIRED_FEATURES].copy()

    # 6) Scale Time and Amount using the saved scaler
    try:
        X[["Time", "Amount"]] = scaler.transform(X[["Time", "Amount"]])
    except Exception as e:
        return pd.DataFrame(columns=["Error"]), f"Error during scaling: {e}"

    # 7) Make predictions (convert to numpy array to skip strict name checks)
    try:
        X_array = X.values
        preds = model.predict(X_array)
        proba = model.predict_proba(X_array)[:, 1]  # probability of fraud class = 1
    except Exception as e:
        return pd.DataFrame(columns=["Error"]), f"Error during prediction: {e}"

    # 8) Build output DataFrame with original data + predictions
    output_df = df.copy()
    output_df["fraud_prediction"] = preds
    output_df["fraud_probability"] = proba

    # 9) Summary text
    total = len(output_df)
    fraud_count = int((preds == 1).sum())
    summary = (
        f"Total rows in uploaded file: {original_rows}\n"
        f"Rows scored: {total} (capped at {max_rows})\n"
        f"Predicted FRAUD: {fraud_count}\n"
        f"Predicted LEGIT: {total - fraud_count}"
    )

    return output_df, summary


scrollable_html = """
<div style='max-height: 450px; overflow-y: scroll; border:1px solid #ccc; padding:5px'>
    <gradio-dataframe id="pred_table"></gradio-dataframe>
</div>
"""


# --- Gradio interface ---
demo = gr.Interface(
    fn=predict_from_csv,
    inputs=gr.File(label="Upload CSV with credit-card transactions"),
    outputs=[
        gr.Dataframe(label="Predictions" , max_height=450),
        gr.Textbox(label="Summary", lines=12),
    ],
    title="💳 Credit Card Fraud Detection",
    description=(
        "Upload a CSV with columns like the Kaggle 'creditcard.csv' dataset "
        "(Time, V1–V28, Amount, optionally Class).\n"
        "The app will score up to 10,000 rows and return fraud predictions "
        "and probabilities for each transaction."
    ),
)

if __name__ == "__main__":
    demo.launch()
