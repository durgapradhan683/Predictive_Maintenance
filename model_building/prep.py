# for data manipulation
import pandas as pd
# for creating a folder
import os
# for data preprocessing and pipeline creation
from sklearn.model_selection import train_test_split
# for hugging face space authentication to upload files
from huggingface_hub import login, HfApi, hf_hub_download

# Define constants for the dataset and output paths
api = HfApi(token=os.getenv("HF_TOKEN"))

# --- Loading the dataset directly from Hugging Face data space ---
repo_id = "dmpradhan/PredictiveMaintenance"
file_in_repo = "Predictive_Maintenance/data/engine_data.csv"
local_file_path = hf_hub_download(repo_id=repo_id, filename=file_in_repo, repo_type="dataset")

df = pd.read_csv(local_file_path)
print("Dataset loaded successfully.")

# --- Data Cleaning: Outlier Removal using IQR method ---
# Columns identified with outliers from EDA
numerical_cols_with_outliers = [
    'Engine rpm',
    'Lub oil pressure',
    'Fuel pressure',
    'Coolant pressure',
    'lub oil temp',
    'Coolant temp'
]

def remove_outliers_iqr(df_input, col_name):
    Q1 = df_input[col_name].quantile(0.25)
    Q3 = df_input[col_name].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    df_output = df_input[(df_input[col_name] >= lower_bound) & (df_input[col_name] <= upper_bound)]
    return df_output

original_shape = df.shape
for col in numerical_cols_with_outliers:
    df = remove_outliers_iqr(df, col)
print(f"Original DataFrame shape: {original_shape}")
print(f"Cleaned DataFrame shape (after removing outliers): {df.shape}")
# --- End of Data Cleaning ---

# Define the target variable for the classification task
target_col = 'Engine Condition'

# Define target variable
y = df[target_col]

# --- Split the cleaned dataset into training and testing sets, and save them locally ---
X = df.drop(columns = [target_col])
Xtrain, Xtest, ytrain, ytest = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

Xtrain.to_csv("Xtrain.csv",index=False)
Xtest.to_csv("Xtest.csv",index=False)
ytrain.to_csv("ytrain.csv",index=False)
ytest.to_csv("ytest.csv",index=False)

# --- Uploading the prepared datasets back to Hugging Face ---
files = ["Xtrain.csv","Xtest.csv","ytrain.csv","ytest.csv"]

for file_path in files:
    api.upload_file(
        path_or_fileobj=file_path,
        path_in_repo=f"Predictive_Maintenance/data/{file_path.split('/')[-1]}",
        repo_id="dmpradhan/PredictiveMaintenance",
        repo_type="dataset",
    )
