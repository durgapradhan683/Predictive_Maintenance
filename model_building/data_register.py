from huggingface_hub.utils import RepositoryNotFoundError, HfHubHTTPError
from huggingface_hub import HfApi, create_repo
import os


repo_id = "dmpradhan/PredictiveMaintenance"
repo_type = "dataset"

# Initialize API client
api = HfApi(token=os.getenv("HF_TOKEN"))

# Step 1: Check if the space exists
try:
    api.repo_info(repo_id=repo_id, repo_type=repo_type)
    print(f"Space '{repo_id}' already exists. Using it.")
except RepositoryNotFoundError:
    print(f"Space '{repo_id}' not found. Creating new space...") # Fixed: Added closing double quote
    create_repo(repo_id=repo_id, repo_type=repo_type, private=False)
    print(f"Space '{repo_id}' created.")

# Determine the absolute path to the engine_data.csv file
# Assuming the script is run from /content/drive/MyDrive/CapstoneProject/Predictive_Maintenance/model_building
# and engine_data.csv is in /content/drive/MyDrive/CapstoneProject/Predictive_Maintenance/data
local_data_file_path = os.path.abspath(os.path.join(os.getcwd(), '../data/engine_data.csv'))

# Define the desired path for the file within the Hugging Face repository
path_in_repo = "Predictive_Maintenance/data/engine_data.csv"

# Upload the single file to the specified path within the repository
api.upload_file(
    path_or_fileobj=local_data_file_path,
    path_in_repo=path_in_repo,
    repo_id=repo_id,
    repo_type=repo_type,
)
print(f"Uploaded {local_data_file_path} to {repo_id}/{path_in_repo}")
