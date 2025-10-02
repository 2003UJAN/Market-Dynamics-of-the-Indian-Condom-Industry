from huggingface_hub import HfApi, HfFolder
from pathlib import Path
import os

# --- Configuration ---
# IMPORTANT: Replace with your Hugging Face username and desired repository name
HF_USERNAME = "YOUR_HF_USERNAME"
HF_REPO_NAME = "condom-market-predictor" # You can name this whatever you like

# Path to your trained model file
LOCAL_MODEL_PATH = Path("models/market_size_predictor.pkl")

def upload_model_to_hf():
    """
    Uploads the trained model to a new or existing Hugging Face repository.
    """
    print("--- Hugging Face Model Uploader ---")
    
    # 1. Authenticate with Hugging Face
    # Make sure you have run `huggingface-cli login` in your terminal
    # or have your HF_TOKEN environment variable set.
    try:
        token = HfFolder.get_token()
        if token is None:
            print("❌ Hugging Face token not found.")
            print("Please run `huggingface-cli login` in your terminal and enter your token.")
            return
    except Exception as e:
        print(f"An error occurred during authentication: {e}")
        return

    print("✅ Authenticated with Hugging Face successfully.")
    
    # 2. Check if the local model file exists
    if not LOCAL_MODEL_PATH.exists():
        print(f"❌ Error: Model file not found at '{LOCAL_MODEL_PATH}'")
        print("Please make sure you have trained the model and it's saved in the correct location.")
        return
        
    print(f"Found model file at: '{LOCAL_MODEL_PATH}'")

    # 3. Create an API client and repository
    api = HfApi()
    repo_id = f"{HF_USERNAME}/{HF_REPO_NAME}"
    
    print(f"Preparing to upload to repository: '{repo_id}'")
    
    api.create_repo(
        repo_id=repo_id,
        repo_type="model",
        exist_ok=True, # Don't raise an error if the repo already exists
    )
    
    # 4. Upload the model file
    try:
        print("Uploading model... this may take a moment for large files.")
        api.upload_file(
            path_or_fileobj=str(LOCAL_MODEL_PATH),
            path_in_repo="market_size_predictor.pkl", # The name of the file in the repo
            repo_id=repo_id,
        )
        print("\n🎉 Success! Your model has been uploaded to the Hugging Face Hub.")
        print(f"You can view it here: https://huggingface.co/{repo_id}")
        
    except Exception as e:
        print(f"❌ An error occurred during the upload: {e}")


if __name__ == "__main__":
    upload_model_to_hf()
