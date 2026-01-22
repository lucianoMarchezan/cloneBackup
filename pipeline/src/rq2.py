from src.utils.convert_to_xml import json_to_xml
from src.config import *
 # RQ2 evaluation 

if __name__ == "__main__": 
    if os.path.exists(FINAL_DATASET_RQ2):
        print(f"{FINAL_DATASET_RQ2} already exists. Skipping conversion.")
    else:
        json_to_xml()