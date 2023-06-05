import pandas as pd
from utils.constants import TARGET_DICT
from utils.preprocess_utils import preprocess_text

# Load data
df = pd.read_csv("/kaggle/input/train3-class/train_3class_balanced.csv", sep=",")
df["text"] = preprocess_text(df["text"])

# Length filtering
df['text_len'] = df.text.str.len()
df = df[(df.text_len >= 5)].reset_index(drop=True)

# Label Encoding
df['target_label'] = df['target'].map(TARGET_DICT)


# Export
df.to_csv("/kaggle/working/teknofestupdated/data/processed/data.csv", index=False)