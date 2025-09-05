import os
from huggingface_hub import snapshot_download

# Model name from Hugging Face
model_name = "F-urkan/rStar2-Agent-14B-Q4_0-GGUF"

# Create a folder named after the model (replace '/' with '_' for valid folder name)
folder_name = model_name.replace("/", "_")
os.makedirs(folder_name, exist_ok=True)

print(f"Downloading model {model_name} to folder {folder_name}...")

# Download the entire repository snapshot
snapshot_download(repo_id=model_name, local_dir=folder_name)

print(f"Model downloaded successfully to {folder_name}")
