import requests
import json

# TCIA API base URL
BASE_URL = "https://services.cancerimagingarchive.net/nbia-api/services/v2/"

# Function to get metadata for LIDC-IDRI
def get_tcia_metadata(collection='LIDC-IDRI'):
    # Define the endpoint for fetching metadata
    endpoint = f"getStudies?Collection={collection}"
    url = BASE_URL + endpoint
    
    # Make the request to TCIA API
    response = requests.get(url)
    
    if response.status_code == 200:
        return response.json()  # Return the metadata as JSON
    else:
        print(f"Error: {response.status_code}")
        return None

# Fetch metadata for LIDC-IDRI dataset
metadata = get_tcia_metadata()
if metadata:
    print(json.dumps(metadata, indent=4))  # Pretty print metadata

import requests
import pydicom
from io import BytesIO

# Function to get images from TCIA using SeriesInstanceUID
def get_images(series_uid):
    url = f"https://services.cancerimagingarchive.net/nbia-api/services/v2/getImage?seriesInstanceUID={series_uid}"
    response = requests.get(url)
    
    if response.status_code == 200:
        dicom_data = BytesIO(response.content)
        dicom_image = pydicom.dcmread(dicom_data)
        return dicom_image
    else:
        print(f"Error fetching image for SeriesInstanceUID {series_uid}")
        return None
