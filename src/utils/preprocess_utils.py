import string
import unicodedata
from .constants import INDICATOR_WORDS

remove_punc = str.maketrans("", "", string.punctuation)


SPIECE_UNDERLINE = "▁".encode("utf-8")


def special_token(x, lang):
    """
    Add special token if text contains upper chars.

    ---------
    param x: Text
    param second_hand_terms : List
    param risky_terms : List
    return: Adjusted text
    """
    for term in INDICATOR_WORDS[lang]["second_hand"]:
        if term in x:
            x = x.replace(term, f"^s^^ {term}")
    for term in INDICATOR_WORDS[lang]["risky"]:
        if term in x:
            x = x.replace(term, f"#r## {term}")
    return x


# Word Count Feature
def feature_wordcount(x):
    """
    Count the word in a text using string split() function. If the length condition met, add special token

    ---------
    param x: Text
    return: Adjusted text
    """
    length = len(x.split())
    if length > 1000:
        return "&nr& " + x
    elif length > 1500:
        return "&nr&& " + x
    elif length > 2000:
        return "&nr&&&" + x
    return x


def preprocess_text(df, prevent_bias=1):
    """
    Remove punctuations, prevent the bias by bias level

    ---------
    param textcol: Text
    param prevent_bias: bias level. 2 means fully-unbiased, 1 means casing-unbiased, 0 means none bias prevention mechanism is being executed
    return: Adjusted text
    """

    # Removing punctuations
    # df['text'] = df['text'].apply(lambda x: x.translate(remove_punc))

    # Casing-Unbiased and Fully-Unbiased Flow
    if prevent_bias > 0:
        df["text"] = df["text"].str.lower()

    df["text"] = df.apply(lambda row: special_token(row["text"], row["lang"]), axis=1)

    df["text"] = df["text"].apply(feature_wordcount)

    return df
