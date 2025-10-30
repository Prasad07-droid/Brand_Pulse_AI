import streamlit as st
import pandas as pd
import requests
import time
import os

# --- Page Configuration & Backend URL ---
st.set_page_config(page_title="Brand Pulse AI", page_icon="💡", layout="wide")

# Backend URL ko apne folder structure ke hisab se set karein
# Agar aap backend/ folder se run kar rahe hain, toh ye sahi hai
BACKEND_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:8000")

# --- Session State Initialization (App's memory) ---
if 'job_id' not in st.session_state:
    st.session_state.job_id = None
if 'results' not in st.session_state:
    st.session_state.results = None
if 'brand_analyzed' not in st.session_state:
    st.session_state.brand_analyzed = ""
if 'error' not in st.session_state:
    st.session_state.error = None

# --- Helper Functions ---

def start_analysis_job(brand_name):
    """
    Backend ko request bhejta hai brand analysis start karne ke liye.
    """
    st.session_state.job_id = None
    st.session_state.results = None
    st.session_state.error = None
    st.session_state.brand_analyzed = brand_name
    try:
        start_resp = requests.post(f"{BACKEND_URL}/analyze", params={"brand": brand_name}, timeout=10)
        start_resp.raise_for_status()
        st.session_state.job_id = start_resp.json().get("job_id")
        print(f"Analysis job started with ID: {st.session_state.job_id}")
    except requests.exceptions.RequestException as e:
        st.session_state.error = f"Connection Error: Could not connect to the backend at {BACKEND_URL}. Please ensure it's running. Details: {e}"
        st.error(st.session_state.error)
    except Exception as e:
        st.session_state.error = f"An unexpected error occurred: {e}"
        st.error(st.session_state.error)

def poll_and_get_results():
    """
    Job status ke liye backend ko poll karta hai aur progress bar update karta hai.
    """
    if not st.session_state.job_id:
        return

    progress_container = st.empty()
    
    with progress_container.container():
        st.info(f"Analysis job started for '{st.session_state.brand_analyzed}'. Please wait, fetching progress...")
        progress_bar = st.progress(0, text="Initializing analysis...")

        while True:
            try:
                status_resp = requests.get(f"{BACKEND_URL}/status/{st.session_state.job_id}", timeout=10)
                status_resp.raise_for_status()
                status_data = status_resp.json()
                status = status_data.get("status")

                if status == "running":
                    progress = status_data.get("progress", 0)
                    total = status_data.get("total", 1)
                    percent_complete = int((progress / total) * 100) if total > 0 else 0
                    progress_text = status_data.get("message", f"Analyzing models... ({progress}/{total})")
                    progress_bar.progress(percent_complete, text=progress_text)
                    time.sleep(2)
                elif status == "complete":
                    progress_bar.progress(100, text="Analysis complete! Fetching final results...")
                    result_resp = requests.get(f"{BACKEND_URL}/result/{st.session_state.job_id}", timeout=60)
                    result_resp.raise_for_status()
                    st.session_state.results = result_resp.json()
                    st.session_state.job_id = None
                    progress_container.empty()
                    st.rerun()
                    break
                elif status == "failed":
                    st.session_state.error = f"Analysis failed. Reason: {status_data.get('message', 'Unknown error')}"
                    st.error(st.session_state.error)
                    st.session_state.job_id = None
                    progress_container.empty()
                    break
                else:
                    progress_bar.progress(0, text=status_data.get("message", "Analysis starting..."))
                    time.sleep(1)
            except requests.exceptions.RequestException as e:
                st.session_state.error = f"Connection to backend lost during analysis. Please try again. Details: {e}"
                st.error(st.session_state.error)
                st.session_state.job_id = None
                progress_container.empty()
                break
            except Exception as e:
                st.session_state.error = f"An unexpected error occurred during polling: {e}"
                st.error(st.session_state.error)
                st.session_state.job_id = None
                progress_container.empty()
                break

def display_dashboard(data):
    """
    Main dashboard (Kaggle data) ko display karta hai.
    """
    st.header(f"Analysis Report for: {data.get('brand')}")
    
    models = data.get("models", [])
    if not models:
        st.warning("No models with reviews found for this brand, or analysis returned no data.")
        return

    df = pd.DataFrame(models)

    total_models = len(df)
    total_reviews = df['total_reviews'].sum()
    overall_pos_percent = (df['positive_reviews'].sum() / total_reviews) * 100 if total_reviews > 0 else 0

    col1, col2, col3 = st.columns(3)
    col1.metric("Models Analyzed", f"{total_models}")
    col2.metric("Total Reviews Processed", f"{int(total_reviews):,}")
    col3.metric("Overall Positive Sentiment", f"{overall_pos_percent:.1f}%")

    st.markdown("---")
    st.subheader("Product Performance Overview")
    
    # ... (Aapka filtering aur table display waala code aage waisa hi rahega) ...
    search_term = st.text_input("Search for a specific model:", placeholder="e.g., iPhone 5s", key="search_model_input")
    filtered_df = df[df['model_name'].str.contains(search_term, case=False, na=False)]
    
    if filtered_df.empty:
        st.warning("No models match your current search.")
    else:
        for index, row in filtered_df.iterrows():
            st.markdown(f"#### {row['model_name']}")
            sub_cols = st.columns([2, 5])
            with sub_cols[0]:
                st.metric("Total Reviews", row['total_reviews'])
                st.metric("Positive", row['positive_reviews'])
                st.metric("Negative", row['negative_reviews'])
            with sub_cols[1]:
                st.info(f"**AI Strategic Suggestion:**\n\n{row['suggestion']}")
            st.markdown("---")


# --- Main App Logic ---
st.title("💡 Brand Pulse AI")
st.markdown("Analyze customer reviews from a massive dataset or scrape live phone reviews for instant insights.")

# --- FEATURE 1: ANALYZE BRAND FROM DATABASE ---
st.header("1. Analyze Brand from Database (Kaggle Dataset)")
brand = st.text_input("Enter brand (e.g., Samsung, Apple):", value=st.session_state.brand_analyzed, placeholder="Type a brand and press Analyze", key="brand_input")

if st.button("Analyze Brand", type="primary"):
    if brand.strip():
        start_analysis_job(brand)
        st.rerun()
    else:
        st.warning("Please enter a brand name to start the analysis.")
        st.session_state.results = None

if st.session_state.job_id and not st.session_state.results:
    poll_and_get_results()

if st.session_state.results:
    display_dashboard(st.session_state.results)

st.markdown("---")

# [--- NAYA CODE START ---]
# --- FEATURE 2: ANALYZE LIVE PHONE REVIEWS ---
st.header("2. Analyze Live Phone Reviews (from GSMArena)")
live_phone_name = st.text_input("Enter Phone Name (e.g., iPhone 15, Samsung S24 Ultra):", placeholder="Type a phone name", key="live_phone_input")

if st.button("Analyze Live Phone"):
    if live_phone_name.strip():
        with st.spinner(f"Scraping live reviews for '{live_phone_name}' from GSMArena... This may take 10-20 seconds..."):
            try:
                # Naye endpoint ko call karo
                live_response = requests.post(
                    f"{BACKEND_URL}/analyze-live-phone",
                    json={"phone_name": live_phone_name},
                    timeout=60 # Scraping + AI mein time lag sakta hai
                )
                live_response.raise_for_status()
                
                live_data = live_response.json()
                
                # Live result ko seedha dikhao
                st.success("Live Analysis Complete!")
                
                st.subheader(f"Live Report for: `{live_data['model_name']}`")
                
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Total Live Reviews", live_data['total_reviews'])
                col2.metric("Positive", live_data['positive_reviews'])
                col3.metric("Negative", live_data['negative_reviews'])
                col4.metric("Neutral", live_data['neutral_reviews'])
                
                st.markdown("---")
                st.subheader("AI-Generated Strategic Suggestion")
                st.info(live_data['suggestion'])

            except requests.exceptions.HTTPError as e:
                # Server se mila error (jaise 404 - No reviews found)
                try:
                    detail = e.response.json().get('detail', 'Unknown error')
                except:
                    detail = e.response.text
                st.error(f"Analysis Failed: {detail}")
            except requests.exceptions.RequestException as e:
                # Connection error
                st.error(f"Connection Error: Could not reach backend. Please ensure the backend server is running. Details: {e}")
            except Exception as e:
                st.error(f"An unexpected error occurred: {e}")
    else:
        st.warning("Please enter a phone name.")