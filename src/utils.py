# src/utils.py
import pandas as pd
import os
import json

def load_teacher_marks_csv(csv_path_or_buffer):
    """
    Accepts filepath or an uploaded buffer (Streamlit file_uploader).
    Expected columns: FILENAME, IDENTITY, CORRECTED_ANSWERS_JSON, SCORE
    """
    df = pd.read_csv(csv_path_or_buffer)
    # ensure required columns exist
    for c in ['FILENAME','IDENTITY','CORRECTED_ANSWERS_JSON','SCORE']:
        if c not in df.columns:
            df[c] = None
    return df

def load_kaggle_dataset_index(path="/kaggle/input/handwriting-recognition/written_name_train_v2.csv"):
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found")
    return pd.read_csv(path)

def save_json(obj, path):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
