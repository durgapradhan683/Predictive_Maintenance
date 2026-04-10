# for data manipulation
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
# for model training, tuning, and evaluation
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, recall_score
# for model serialization
import joblib
# for creating a folder
import os
# for hugging face space authentication to upload files
from huggingface_hub import login, HfApi, create_repo, hf_hub_download
from huggingface_hub.utils import RepositoryNotFoundError, HfHubHTTPError

# Removed: !pip install mlflow==3.0.1 pyngrok==7.2.12 -q # This will be run in the execution cell
import mlflow

# Start MLflow UI server in the background
# !nohup mlflow ui --host 0.0.0.0 --port 5000 & # This will be run in the execution cell
# !sleep 5 # Give the server some time to start # This will be run in the execution cell

# --- Experimentation Tracking: MLflow Setup ---
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("predictive-maintenance-experiment")

api = HfApi(token=os.getenv("HF_TOKEN"))

# --- Load the train and test data from the Hugging Face data space ---
Xtrain_path = hf_hub_download(repo_id="dmpradhan/PredictiveMaintenance", filename="Predictive_Maintenance/data/Xtrain.csv", repo_type="dataset")
Xtest_path = hf_hub_download(repo_id="dmpradhan/PredictiveMaintenance", filename="Predictive_Maintenance/data/Xtest.csv", repo_type="dataset")
ytrain_path = hf_hub_download(repo_id="dmpradhan/PredictiveMaintenance", filename="Predictive_Maintenance/data/ytrain.csv", repo_type="dataset")
ytest_path = hf_hub_download(repo_id="dmpradhan/PredictiveMaintenance", filename="Predictive_Maintenance/data/ytest.csv", repo_type="dataset")

Xtrain = pd.read_csv(Xtrain_path)
Xtest = pd.read_csv(Xtest_path)
ytrain = pd.read_csv(ytrain_path).iloc[:, 0] # ytrain is read as a DataFrame, need to convert to Series
ytest = pd.read_csv(ytest_path).iloc[:, 0] # ytest is read as a DataFrame, need to convert to Series

# --- Feature Definition (for Preprocessing) ---
# Based on the data description and info, all features are numerical
numeric_features = [
    'Engine rpm',
    'Lub oil pressure',
    'Fuel pressure',
    'Coolant pressure',
    'lub oil temp',
    'Coolant temp'
]
categorical_features = [] # No categorical features in this dataset

# --- Class Weight Calculation ---
# Set the class weight to handle class imbalance
# Check if both classes exist before calculating
if 0 in ytrain.value_counts() and 1 in ytrain.value_counts():
    class_weight = ytrain.value_counts()[0] / ytrain.value_counts()[1]
else:
    # Handle cases where one class might be missing (unlikely in this context but good for robustness)
    class_weight = 1 # No imbalance or can't calculate

# --- Preprocessing Steps ---
# Define the preprocessing steps
# Only StandardScaler for numeric features, OneHotEncoder will be empty as no categorical features
preprocessor = make_column_transformer(
    (StandardScaler(), numeric_features),
    (OneHotEncoder(handle_unknown='ignore'), categorical_features)
)

# --- Define a model and parameters (XGBoost Classifier) ---
xgb_model = xgb.XGBClassifier(scale_pos_weight=class_weight, random_state=42)

# Define hyperparameter grid for GridSearchCV
param_grid = {
    'xgbclassifier__n_estimators': [100, 200],
    'xgbclassifier__max_depth': [3, 5],
    'xgbclassifier__colsample_bytree': [0.5, 0.7],
    'xgbclassifier__colsample_bylevel': [0.5, 0.7],
    'xgbclassifier__learning_rate': [0.05, 0.1],
    'xgbclassifier__reg_lambda': [0.5, 1]
}

# Model pipeline
model_pipeline = make_pipeline(preprocessor, xgb_model)

# Start MLflow run for Experimentation Tracking
with mlflow.start_run():
    # --- Tune the model with the defined parameters using GridSearchCV ---
    grid_search = GridSearchCV(model_pipeline, param_grid, cv=5, n_jobs=-1)
    grid_search.fit(Xtrain, ytrain)

    # --- Log all the tuned parameters to MLflow ---
    results = grid_search.cv_results_
    for i in range(len(results['params'])):
        param_set = results['params'][i]
        mean_score = results['mean_test_score'][i]
        std_score = results['std_test_score'][i]

        # Log each combination as a separate MLflow run
        with mlflow.start_run(nested=True):
            mlflow.log_params(param_set)
            mlflow.log_metric("mean_test_score", mean_score)
            mlflow.log_metric("std_test_score", std_score)

    # Log best parameters separately in main run
    mlflow.log_params(grid_search.best_params_)

    # Store and evaluate the best model
    best_model = grid_search.best_estimator_

    classification_threshold = 0.5 # Default threshold for binary classification

    y_pred_train_proba = best_model.predict_proba(Xtrain)[:, 1]
    y_pred_train = (y_pred_train_proba >= classification_threshold).astype(int)

    y_pred_test_proba = best_model.predict_proba(Xtest)[:, 1]
    y_pred_test = (y_pred_test_proba >= classification_threshold).astype(int)

    # --- Evaluate the model performance ---
    train_report = classification_report(ytrain, y_pred_train, output_dict=True)
    test_report = classification_report(ytest, y_pred_test, output_dict=True)

    # Log the metrics for the best model
    mlflow.log_metrics({
        "train_accuracy": train_report['accuracy'],
        "train_precision": train_report['1']['precision'],
        "train_recall": train_report['1']['recall'],
        "train_f1-score": train_report['1']['f1-score'],
        "test_accuracy": test_report['accuracy'],
        "test_precision": test_report['1']['precision'],
        "test_recall": test_report['1']['recall'],
        "test_f1-score": test_report['1']['f1-score']
    })

    # --- Model Storage (Local) ---
    model_path = "best_PredictiveMaintenance_model_v1.joblib"
    joblib.dump(best_model, model_path)

    # Log the model artifact to MLflow
    mlflow.log_artifact(model_path, artifact_path="model")
    print(f"Model saved as artifact at: {model_path}")

    # --- Register the best model in the Hugging Face model hub ---
    repo_id = "dmpradhan/PredictiveMaintenance"
    repo_type = "model"

    # Step 1: Check if the space exists
    try:
        api.repo_info(repo_id=repo_id, repo_type=repo_type)
        print(f"Space '{repo_id}' already exists. Using it.")
    except RepositoryNotFoundError:
        print(f"Space '{repo_id}' not found. Creating new space...")
        create_repo(repo_id=repo_id, repo_type=repo_type, private=False)
        print(f"Space '{repo_id}' created.")

    api.upload_file(
        path_or_fileobj="best_PredictiveMaintenance_model_v1.joblib",
        path_in_repo="best_PredictiveMaintenance_model_v1.joblib",
        repo_id=repo_id,
        repo_type=repo_type,
    )
