import os
import zipfile
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords

# UNZIPING DATASETS
data_dir = "data"
extract_dirs = []

#Extracting the zip file into its respective folder
for file in os.listdir(data_dir):
    if file.endswith(".zip"):
        folder_name = os.path.join(data_dir, file.replace(".zip", ""))
        extract_dirs.append(folder_name)
        with zipfile.ZipFile(os.path.join(data_dir, file), 'r') as zip_ref:
            zip_ref.extractall(folder_name)
            print(f"Extracted: {file}")


# LOADING ALL CSV FILES
def load_csvs_from_folder(folder):
    dfs = []
    for root, dirs, files in os.walk(folder):
        for f in files:
            if f.endswith(".csv"):
                try:
                    df = pd.read_csv(os.path.join(root, f))
                    dfs.append(df)
                except Exception as e:
                    print(f"Error reading {f}: {e}")
    return dfs

all_dfs = []

for path in extract_dirs:
    dfs = load_csvs_from_folder(path)
    all_dfs.extend(dfs)

# Adding the stand alone dataset
csv_path = os.path.join(data_dir, "bug_severity_data.csv")
if os.path.exists(csv_path):
    all_dfs.append(pd.read_csv(csv_path))

print(f"Total files loaded: {len(all_dfs)}")


# STANDARDIZING COLUMNS BEFORE MERGING
def standardize_columns(df):
    df = df.rename(columns={
        'summary': 'description',
        'short_desc': 'description',
        'bug_report': 'description',
        'text': 'description',
        'desc': 'description',
        'severity_level': 'severity',
        'bug_severity': 'severity',
        'class': 'severity'
    })
    keep_cols = [c for c in ['description', 'severity'] if c in df.columns]
    return df[keep_cols].dropna(subset=keep_cols)

#Applying to all data frames
std_dfs = [standardize_columns(df) for df in all_dfs if not df.empty]


# MERGING ALL DATASETS
merged = pd.concat(std_dfs, ignore_index=True)
merged.drop_duplicates(subset=['description'], inplace=True)
print("Merged shape:", merged.shape)


#DATA PREPROCESSING :

# CLEANING TEXT
nltk.download('stopwords')
stop = set(stopwords.words('english'))

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+", "", text) #Remove URLs
    text = re.sub(r"[^a-z\s]", "", text) #Remove punctuation, numbers
    return " ".join(w for w in text.split() if w not in stop)

merged['cleaned_desc'] = merged['description'].apply(clean_text)


# NORMALIZING SEVERITY LABELS
severity_map = {
    'blocker': 'critical', 'critical': 'critical', 'high': 'critical',
    'major': 'major', 'normal': 'minor',
    'minor': 'minor', 'trivial': 'trivial', 'low': 'trivial',
    'enhancement': 'trivial', 'unknown': 'minor'
}
merged['severity'] = merged['severity'].astype(str).str.lower().map(severity_map)
merged = merged.dropna(subset=['severity'])

# SAVING FINAL MERGED FILE
merged.to_csv("merged_dataset.csv", index=False)
print("✅ Saved merged_dataset.csv successfully!")

