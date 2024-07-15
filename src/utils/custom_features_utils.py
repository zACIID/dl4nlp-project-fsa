import os
import requests
from zipfile import ZipFile


def download_and_extract_zip(url: str, extract_to: str):
    """
    Download and extract a zip (extract all files)

    :param url: The URL to download the zip file from.
    :param extract_to: The directory to extract the zip file to.
    """
    zip_path = os.path.join(extract_to, 'dataset.zip')

    # Create directory if it doesn't exist
    os.makedirs(extract_to, exist_ok=True)

    # Download the zip file if it doesn't exist
    if not os.path.exists(zip_path):
        response = requests.get(url)
        with open(zip_path, 'wb') as f:
            f.write(response.content)
        print(f"Zip file downloaded and saved to {zip_path}")
    else:
        print(f"Zip file already exists at {zip_path}")

    # Extract all files from the zip archive
    with ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
        print(f"Files extracted to {extract_to}")
