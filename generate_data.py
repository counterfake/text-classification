import pandas as pd
from utils.constants import TARGET_DICT, TARGET_DICT_FASHION
from utils.preprocess_utils import preprocess_text
from sklearn.model_selection import StratifiedKFold
from sklearn.utils import shuffle
import argparse

parser = argparse.ArgumentParser(
    description="Generating preprocessed text classification data"
)
parser.add_argument(
    "--data_path",
    required=True,
    type=str,
    help="Path to pandas-readable not-validation splitted dataset file.",
)
parser.add_argument("--data_name", required=True, type=str, help="name of the data")
parser.add_argument(
    "--fashion", type=bool, default=False, help="Is it category prediction"
)

opt = parser.parse_args()  # Corrected method

# Load data
df = pd.read_csv(opt.data_path, sep=",")


if opt.fashion:
    df["text"] = df["title"]
else:
    df["text"] = df["description_text"]
    preprocess_text(df)

# Length filtering
df["text_len"] = df.text.str.len()
df = df[(df.text_len >= 5)].reset_index(drop=True)

# Label Encoding
if opt.fashion:
    df["target"] = df["related_product"].map(TARGET_DICT_FASHION)
else:
    df["target"] = df["category"].map(TARGET_DICT)

# Shuffle the DataFrame
df_shuffled = shuffle(df, random_state=42)


# Export shuffled DataFrame
df_shuffled.to_csv(f"../data/processed/{opt.data_name}.csv", index=False)
