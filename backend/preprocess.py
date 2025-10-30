import pandas as pd
import re
import string
from typing import List
import numpy as np

# NLTK imports for enhanced cleaning
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# --- Ensure NLTK data is available (helpful for first-time setup) ---
try:
    STOPWORDS = set(stopwords.words('english'))
except LookupError:
    print("NLTK stopwords not found. Downloading...")
    nltk.download('stopwords')
    STOPWORDS = set(stopwords.words('english'))

try:
    lemmatizer = WordNetLemmatizer()
    lemmatizer.lemmatize("test") # Just to trigger download if not present
except LookupError:
    print("NLTK WordNet not found. Downloading...")
    nltk.download('wordnet')

def load_data(path: str) -> pd.DataFrame:
    """
    Load cleaned CSV (should contain Brand Name, Product Name, Rating, Reviews).
    Returns a DataFrame with 'sentiment' column derived from Rating.
    """
    df = pd.read_csv(path)
    required = ['Brand Name', 'Product Name', 'Rating', 'Reviews']
    for c in required:
        if c not in df.columns:
            raise ValueError(f"Missing column {c} in dataset")

    # Ensure consistent data types
    df['Brand Name'] = df['Brand Name'].astype(str)
    df['Product Name'] = df['Product Name'].astype(str)
    df['Reviews'] = df['Reviews'].astype(str)

    ratings_numeric = pd.to_numeric(df['Rating'], errors='coerce')
    
    # Define sentiment based on ratings
    conditions = [
        ratings_numeric >= 4,  # 4 and 5 stars are positive
        ratings_numeric == 3,  # 3 stars are neutral
        ratings_numeric < 3    # 1 and 2 stars are negative
    ]
    choices = ['positive', 'neutral', 'negative']
    
    df['sentiment'] = np.select(conditions, choices, default='neutral')
    df = df.dropna(subset=['Reviews', 'Brand Name', 'Product Name']).reset_index(drop=True)
    return df

def simple_clean(text: str) -> str:
    """
    Light cleaning for tokenizer input.
    THIS FUNCTION MUST MATCH THE ONE USED FOR TRAINING THE CURRENT MODEL.
    """
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text, flags=re.MULTILINE)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def enhanced_clean(text: str) -> str:
    """
    More thorough text cleaning including stop word removal and lemmatization.
    NOTE: If you use this, you MUST retrain your model with data cleaned by this function.
    """
    if not isinstance(text, str):
        return ""
    
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text, flags=re.MULTILINE)
    text = text.translate(str.maketrans("", "", string.punctuation))
    
    tokens = text.split()
    tokens = [word for word in tokens if word not in STOPWORDS]
    
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    
    return " ".join(tokens)