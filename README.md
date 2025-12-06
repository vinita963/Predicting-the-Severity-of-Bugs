# Predicting-the-Severity-of-Bugs

**M.Tech Final Year Project**  
Use of **NLP-based Transformer Models** for automatic bug severity prediction.

This project aims to predict the **severity level of software bug reports** using Natural Language Processing and Machine Learning techniques, helping developers prioritize critical issues efficiently.

---

##  Project Overview

Software bug tracking systems generate large volumes of bug reports. Manually identifying the severity of each bug is time-consuming and error-prone.  
This project automates **bug severity prediction** by analyzing the textual description of bug reports using NLP and transformer-based models.

---
### Technologies Used

Python,
NLP,
Transformer Models,
Scikit-learn,
HuggingFace Transformers,
SHAP (Explainable AI),
Streamlit

##  Key Features

- Bug severity prediction from textual descriptions
- NLP preprocessing and feature extraction
- Transformer-based model training and inference
- Explainable AI using SHAP and sentence-level explanations
- Streamlit-based user interface

---

##  Project Structure

Predicting-the-Severity-of-Bugs/
│
├── src/
│ ├── app.py
│ ├── data.py
│ ├── train_model.py
│ ├── predict.py
│ ├── infer_explain.py
│ ├── sentence_explain.py
│ └── shap_explain.py
│
├── README.md
├── requirements.txt
└── .gitignore


---

##  Datasets

This project uses **three software bug report datasets** for training and evaluation:

- **BugHub Dataset**
- **Unified Bug Dataset**
- **Merged Dataset** (used for final training)

 Due to GitHub size limitations, datasets are **not included in this repository**.

---

##  Dataset Access (Google Drive)

All datasets are hosted on Google Drive.

 **Download Datasets:**  
https://drive.google.com/drive/folders/1qNhAQggXFHRENAUcWP7ilFlm98pSOtEx?usp=sharing

After downloading, place the files in the following structure:
data/
├── BugHub/
├── UnifiedBugDataset/
└── merged_dataset.csv


---

##  Installation & Setup

Step 1: Create virtual environment (optional but recommended)

python -m venv venv
venv\Scripts\activate        # Windows

Step 2: Install dependencies
pip install -r requirements.txt

Step 3: Model Training

To train the bug severity prediction model:
python src/train_model.py

Step 4: Run the Application

To start the Streamlit application:
streamlit run src/app.py


