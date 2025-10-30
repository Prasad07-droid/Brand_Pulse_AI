import os
import json
import time
import pickle
from typing import List, Dict
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from fastapi import FastAPI, BackgroundTasks, HTTPException
from pydantic import BaseModel
import asyncio
from gsmarena_scraper_selenium import scrape_gsmarena_reviews_selenium

from preprocess import load_data, simple_clean
from suggestion import generate_nlp_suggestion

# [--- SELENIUM CHANGE: Naya scraper import karo ---]
try:
    # Naya Selenium scraper import karo
    from gsmarena_scraper_selenium import scrape_gsmarena_reviews_selenium
    print("GSMArena SELENIUM scraper loaded successfully.")
except ImportError:
    print("WARNING: gsmarena_scraper_selenium.py not found. Live analysis will not work.")
    def scrape_gsmarena_reviews_selenium(phone_name: str, max_pages: int = 1):
        return ["Error: gsmarena_scraper_selenium.py file is missing."]
# [--- SELENIUM CHANGE END ---]


# ===================================================================
# CONFIGURATION
# ===================================================================
BASE_DIR = os.path.dirname(__file__)
DATA_PATH = os.path.join(BASE_DIR, 'data', 'cleaned_brands.csv')
MODEL_DIR = os.path.join(BASE_DIR, 'model')
TOKENIZER_PATH = os.path.join(MODEL_DIR, 'tokenizer.pkl')
MODEL_PATH = os.path.join(MODEL_DIR, 'lstm_model.keras')

# --- TensorFlow Keras setup ---
from tensorflow import keras
try:
    with open(TOKENIZER_PATH, 'rb') as f:
        tokenizer = pickle.load(f)
    sentiment_model = keras.models.load_model(MODEL_PATH)
    print("Sentiment model and tokenizer loaded successfully.")
except Exception as e:
    tokenizer = None
    sentiment_model = None
    print(f"FATAL: Could not load sentiment model or tokenizer. Error: {e}")
    print("Please ensure 'model_train.py' was run and model files exist.")

# --- Data Loading ---
try:
    df_all_reviews = load_data(DATA_PATH)
    print("All review data loaded successfully.")
except Exception as e:
    df_all_reviews = pd.DataFrame()
    print(f"FATAL: Could not load review data. Error: {e}")
    print(f"Please ensure '{DATA_PATH}' exists and is correctly formatted.")

MAXLEN = 120

# ===================================================================
# FastAPI App Initialization
# ===================================================================
app = FastAPI(
    title="Brand Pulse AI Backend",
    description="API for sentiment analysis and strategic suggestion generation.",
    version="1.0.0",
)

job_status: Dict[str, Dict] = {}
job_results: Dict[str, Dict] = {}

# ===================================================================
# Helper Functions for Sentiment Prediction
# ===================================================================
def predict_sentiment_batch(texts: List[str]) -> List[str]:
    """Predicts sentiment for a list of texts."""
    if tokenizer is None or sentiment_model is None:
        return ["error"] * len(texts)

    cleaned_texts = [simple_clean(text) for text in texts]
    sequences = tokenizer.texts_to_sequences(cleaned_texts)
    padded_sequences = pad_sequences(sequences, maxlen=MAXLEN, padding='post')
    predictions = sentiment_model.predict(padded_sequences, verbose=0)

    sentiments = []
    for pred in predictions:
        if np.argmax(pred) == 1:
            sentiments.append('positive')
        else:
            sentiments.append('negative')
    return sentiments

# ===================================================================
# Background Task for Analysis (Kaggle Dataset)
# ===================================================================
async def run_analysis_task(job_id: str, brand_name: str):
    """Performs the full sentiment analysis and suggestion generation for a brand."""
    job_status[job_id] = {"status": "running", "progress": 0, "total": 1, "message": "Starting analysis..."}

    if df_all_reviews.empty:
        job_status[job_id].update({"status": "failed", "message": "Review data not loaded."})
        return

    brand_df = df_all_reviews[df_all_reviews['Brand Name'].str.contains(brand_name, case=False, na=False)].copy()
    if brand_df.empty:
        job_status[job_id].update({"status": "complete", "progress": 1, "total": 1, "message": "No products found for this brand."})
        job_results[job_id] = {"brand": brand_name, "models": []}
        return

    product_models = brand_df['Product Name'].unique()
    total_models = len(product_models)
    models_data = []

    for i, model_name in enumerate(product_models):
        # ... (rest of the loop code remains the same as before) ...
        if job_status[job_id].get("status") == "failed":
            break

        model_reviews_df = brand_df[brand_df['Product Name'] == model_name]
        reviews_for_prediction = model_reviews_df['Reviews'].tolist()
        if not reviews_for_prediction:
            continue

        predicted_sentiments = predict_sentiment_batch(reviews_for_prediction)
        original_sentiments = model_reviews_df['sentiment'].tolist()

        positive_reviews_list = [
            review for review, original_s, predicted_s in zip(reviews_for_prediction, original_sentiments, predicted_sentiments)
            if original_s == 'positive' or predicted_s == 'positive'
        ]
        negative_reviews_list = [
            review for review, original_s, predicted_s in zip(reviews_for_prediction, original_sentiments, predicted_sentiments)
            if original_s == 'negative' or predicted_s == 'negative'
        ]

        total_reviews = len(reviews_for_prediction)
        positive_count = len(positive_reviews_list)
        negative_count = len(negative_reviews_list)
        neutral_count = total_reviews - (positive_count + negative_count)

        suggestion_text = await asyncio.to_thread(
            generate_nlp_suggestion,
            model_name,
            positive_reviews_list,
            negative_reviews_list,
            positive_count,
            negative_count
        )

        models_data.append({
            "model_name": model_name,
            "total_reviews": total_reviews,
            "positive_reviews": positive_count,
            "neutral_reviews": neutral_count,
            "negative_reviews": negative_count,
            "suggestion": suggestion_text
        })

        job_status[job_id].update({
            "progress": i + 1,
            "total": total_models,
            "message": f"Analyzing {model_name}..."
        })
        await asyncio.sleep(0.1)

    job_results[job_id] = {"brand": brand_name, "models": models_data}
    job_status[job_id].update({"status": "complete", "progress": total_models, "message": "Analysis complete."})

# ===================================================================
# API Endpoints (Feature 1: Kaggle Dataset)
# ===================================================================

class AnalysisRequest(BaseModel):
    brand: str

@app.post("/analyze")
async def analyze_brand(brand: str, background_tasks: BackgroundTasks):
    """Starts the analysis for a given brand from the dataset."""
    if sentiment_model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="AI models not loaded.")
    if df_all_reviews.empty:
        raise HTTPException(status_code=503, detail="Review data not loaded.")

    job_id = str(time.time())
    job_status[job_id] = {"status": "starting", "progress": 0, "total": 1, "message": "Initializing..."}
    background_tasks.add_task(run_analysis_task, job_id, brand)
    return {"job_id": job_id, "message": f"Analysis started for brand: {brand}"}

@app.get("/status/{job_id}")
async def get_analysis_status(job_id: str):
    """Returns the status of an analysis job."""
    status = job_status.get(job_id)
    if not status:
        raise HTTPException(status_code=404, detail="Job ID not found.")
    return status

@app.get("/result/{job_id}")
async def get_analysis_result(job_id: str):
    """Returns the results of a completed analysis job."""
    status = job_status.get(job_id)
    if not status:
        raise HTTPException(status_code=404, detail="Job ID not found.")
    if status["status"] != "complete":
        raise HTTPException(status_code=400, detail="Job is not yet complete.")
    result = job_results.get(job_id)
    if not result:
        raise HTTPException(status_code=404, detail="Results not found.")
    return result

# ===================================================================
# API Endpoints (Feature 2: Live GSMArena Analysis with Selenium)
# ===================================================================

class LivePhoneRequest(BaseModel):
    phone_name: str

@app.post("/analyze-live-phone")
async def analyze_live_phone(request: LivePhoneRequest):
    """Analyzes live reviews from GSMArena using Selenium."""
    if sentiment_model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="AI models not loaded.")

    try:
        phone_name = request.phone_name
        print(f"Starting live analysis for: {phone_name} using Selenium scraper")

        # [--- SELENIUM CHANGE: Call the Selenium scraper function ---]
        # Run the potentially slow I/O-bound scraping in a separate thread
        live_reviews_list = await asyncio.to_thread(
            scrape_gsmarena_reviews_selenium, # <-- Use the Selenium function
            phone_name,
            max_pages=3  # Selenium slow hai, kam pages scrape karo
        )
        # [--- SELENIUM CHANGE END ---]

        # Check if scraper returned an error message
        if not live_reviews_list or (isinstance(live_reviews_list, list) and live_reviews_list[0].startswith("Error:")):
            error_detail = live_reviews_list[0] if live_reviews_list else "Unknown scraping error."
            print(f"Scraping failed: {error_detail}")
            raise HTTPException(status_code=404, detail=f"Could not scrape reviews for '{phone_name}'. Reason: {error_detail}")

        print(f"Successfully scraped {len(live_reviews_list)} reviews using Selenium.")

        # --- Process reviews (same as before) ---
        predicted_sentiments = predict_sentiment_batch(live_reviews_list)
        positive_reviews_list = [r for r, s in zip(live_reviews_list, predicted_sentiments) if s == 'positive']
        negative_reviews_list = [r for r, s in zip(live_reviews_list, predicted_sentiments) if s == 'negative']
        pos_count = len(positive_reviews_list)
        neg_count = len(negative_reviews_list)
        total_reviews = len(live_reviews_list)
        neutral_count = total_reviews - (pos_count + neg_count)

        # --- Generate suggestion (same as before) ---
        suggestion_text = await asyncio.to_thread(
            generate_nlp_suggestion,
            phone_name,
            positive_reviews_list,
            negative_reviews_list,
            pos_count,
            neg_count
        )

        # --- Return results (same as before) ---
        return {
            "model_name": f"Live Analysis (Selenium): {phone_name}",
            "total_reviews": total_reviews,
            "positive_reviews": pos_count,
            "neutral_reviews": neutral_count,
            "negative_reviews": neg_count,
            "suggestion": suggestion_text
        }

    except HTTPException as http_exc:
        # Re-raise HTTP exceptions directly
        raise http_exc
    except Exception as e:
        # Catch any other unexpected errors during the process
        print(f"Unexpected error during live analysis for {request.phone_name}: {e}")
        # import traceback # Uncomment for detailed debug logs
        # print(traceback.format_exc()) # Uncomment for detailed debug logs
        raise HTTPException(status_code=500, detail=f"An internal error occurred during live analysis: {str(e)}")


# Health check endpoint
@app.get("/")
async def read_root():
    return {"message": "Brand Pulse AI Backend is running!"}