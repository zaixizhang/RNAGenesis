import requests
import json
import os

# Configuration
TOKEN = "DRIEnfFVjvbbgZmIxf00kvBFU941SHpgIbscYzvxknGlWqksmPrOZxi70Yp9"  # Replace with your Zenodo API token
FILE_PATH = r"C:\Users\Jrf51\OneDrive\Desktop\RNAGenesis\checkpoints.zip"  # Path to your file
ZENODO_URL = "https://zenodo.org/api/deposit/depositions"

# Step 1: Create a new deposition
headers = {"Authorization": f"Bearer {TOKEN}"}
response = requests.post(ZENODO_URL, params={"access_token": TOKEN}, json={})
if response.status_code != 201:
    print(f"Error creating deposition: {response.text}")
    exit(1)
deposition = response.json()
deposition_id = deposition["id"]
bucket_url = deposition["links"]["bucket"]

# Step 2: Upload the file
file_name = os.path.basename(FILE_PATH)
print(f"Uploading {file_name}...")
with open(FILE_PATH, "rb") as f:
    response = requests.put(f"{bucket_url}/{file_name}", data=f, headers=headers)
if response.status_code != 200:
    print(f"Error uploading file: {response.text}")
    exit(1)

# Step 3: Set metadata
metadata = {
    "metadata": {
        "title": "checkpoints for RNAGenesis",
        "upload_type": "dataset",  # or "software" if it's code/models
        "description": (
            "This dataset contains model checkpoints for RNAGenesis. "
            "See README.md for more instructions."
            "MD5: B9EBCBDBFB856D65C9B290006E21C03D"
        ),
        "creators": [
            {
                "name": "Jin, Ruofan",
                "affiliation": "Zhejiang University",
                "orcid": "0009-0003-8104-7643"  # Optional, remove if no ORCID
            }
        ],
        "access_right": "open",  # Use "embargo" for delayed release
        "license": "cc-by-4.0",  # Common for academic data
        "keywords": ["RNA", "foundation model", "genomics", "checkpoints"]
    }
}
response = requests.put(
    f"{ZENODO_URL}/{deposition_id}",
    headers={**headers, "Content-Type": "application/json"},
    json=metadata
)
if response.status_code != 200:
    print(f"Error setting metadata: {response.text}")
    exit(1)

# Step 4: Publish
response = requests.post(f"{ZENODO_URL}/{deposition_id}/actions/publish", headers=headers)
if response.status_code != 202:
    print(f"Error publishing: {response.text}")
    exit(1)
doi = response.json()["doi"]
print(f"Success! Published at: https://doi.org/{doi}")