# app.py
import streamlit as st
from src.ocr import extract_text_from_image_batch
from src.analysis import phase1_generate_reports, phase2_auto_check
from src.utils import load_teacher_marks_csv, load_kaggle_dataset_index
import pandas as pd
import os

st.set_page_config(page_title="Handwritten Assessment Tool", layout="wide")

st.title("AI Exam Assessment — Handwritten (Prototype)")

st.sidebar.header("Mode")
mode = st.sidebar.radio("Choose phase", ["Phase I — Teacher-assisted", "Phase II — Automated"])

st.sidebar.markdown("**Dataset paths** (optional – used to pre-load examples)")
kaggle_index = st.sidebar.text_input("Kaggle CSV path", value="/kaggle/input/handwriting-recognition/written_name_train_v2.csv")
kaggle_images = st.sidebar.text_input("Kaggle images folder", value="/kaggle/input/handwriting-recognition/train_v2/train")

st.header("1) Upload scanned answer sheets (images / PDFs)")
uploaded_files = st.file_uploader("Upload multiple images or PDFs", accept_multiple_files=True, type=['png','jpg','jpeg','tif','tiff','pdf'])

st.header("2) Teacher marks / gold data (CSV)")
marks_file = st.file_uploader("Upload teacher-checked CSV (columns: FILENAME, IDENTITY, CORRECTED_ANSWERS_JSON, SCORE)", type=['csv'])

if st.button("Load example Kaggle index (demo)"):
    try:
        df_index = load_kaggle_dataset_index(kaggle_index)
        st.success("Loaded index — showing first 10 rows")
        st.dataframe(df_index.head(10))
    except Exception as e:
        st.error(f"Failed to load index: {e}")

if uploaded_files:
    st.info(f"{len(uploaded_files)} files uploaded — extracting text (OCR). This may take a moment.")
    # Save temp files
    tmp_dir = "tmp_uploads"
    os.makedirs(tmp_dir, exist_ok=True)
    imgs = []
    for f in uploaded_files:
        path = os.path.join(tmp_dir, f.name)
        with open(path, "wb") as out:
            out.write(f.getbuffer())
        imgs.append(path)

    ocr_results = extract_text_from_image_batch(imgs)
    st.subheader("OCR Results (first 3)")
    for i, (fname, text) in enumerate(ocr_results.items()):
        if i >= 3: break
        st.write(f"**{fname}**")
        st.code(text[:1000] + ("..." if len(text)>1000 else ""))
else:
    ocr_results = {}

if marks_file:
    teacher_df = load_teacher_marks_csv(marks_file)
    st.success("Teacher marks loaded")
    st.dataframe(teacher_df.head(10))
else:
    teacher_df = None

if st.button("Run Analysis"):
    if mode.startswith("Phase I"):
        if teacher_df is None or not ocr_results:
            st.error("Upload both scanned files and teacher marks to run Phase I.")
        else:
            reports = phase1_generate_reports(ocr_results, teacher_df)
            st.success("Phase I analysis complete — reports generated.")
            for sid, report in reports.items():
                st.subheader(f"Student: {sid}")
                st.json(report)
            # allow download
            df_reports = pd.DataFrame.from_dict(reports, orient='index')
            st.download_button("Download reports (CSV)", df_reports.to_csv(index=True).encode('utf-8'), file_name="phase1_reports.csv")
    else:
        # Phase II
        if not ocr_results:
            st.error("Upload scanned files to run Phase II.")
        else:
            # For demo, we require an answer key in sidebar (JSON string)
            answer_key_str = st.text_area("Put answer key (JSON) for MCQs/short answers", value='{"Q1":"42","Q2":["keyword1","keyword2"], "MCQ":{"Q3":"B"}}')
            reports = phase2_auto_check(ocr_results, answer_key_str)
            st.success("Phase II auto-check complete.")
            for sid, report in reports.items():
                st.subheader(f"Student: {sid}")
                st.json(report)
            df_reports = pd.DataFrame.from_dict(reports, orient='index')
            st.download_button("Download phase2_reports.csv", df_reports.to_csv(index=True).encode('utf-8'), file_name="phase2_reports.csv")

st.sidebar.markdown("---")
st.sidebar.markdown("Prototype — replace heuristics with trained models for production.")
