import os
from dotenv import load_dotenv

# .env file se environment variables load karein
load_dotenv()
import os
from groq import Groq
from typing import List, Dict
import json

# ===================================================================
# GROQ API CONFIGURATION
# ===================================================================
# IMPORTANT: Replace "YOUR_GROQ_API_KEY_HERE" with your actual Groq API key
# It's highly recommended to set this as an environment variable for production.
try:
    api_key = os.getenv("GROQ_API_KEY")
    print("Groq client initialized successfully.")
except Exception as e:
    client = None
    print(f"FATAL: Could not initialize Groq client. Error: {e}")

# The latest, stable model available on Groq
MODEL_NAME = "allam-2-7b" # Or "llama-3.1-70b-versatile" for more complex tasks if needed
# ===================================================================

# --- Caching setup ---
CACHE_FILE = os.path.join(os.path.dirname(__file__), 'suggestion_cache.json')

def load_cache() -> Dict:
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, 'r') as f:
            return json.load(f)
    return {}

def save_cache(cache: Dict):
    with open(CACHE_FILE, 'w') as f:
        json.dump(cache, f, indent=4)

def generate_nlp_suggestion(model_name: str, positive_reviews: List[str], negative_reviews: List[str], pos_count: int, neg_count: int) -> str:
    """
    Generates high-quality, complete strategic suggestions using the Groq API.
    Uses caching to avoid redundant API calls.
    """
    # Create a unique cache key for this model's data
    cache_key = f"{model_name}-{pos_count}-{neg_count}"
    cache = load_cache()

    if cache_key in cache:
        print(f"Serving suggestion for {model_name} from cache.")
        return cache[cache_key]

    if client is None:
        return "Groq AI client is not available. Please check the server logs for errors."
    if not positive_reviews and not negative_reviews:
        return "Not enough review data to generate a strategic suggestion."

    # Take a reasonable sample of reviews for the LLM prompt
    pos_sample = ". ".join(positive_reviews[:10]) # Limit to 10 reviews
    neg_sample = ". ".join(negative_reviews[:10]) # Limit to 10 reviews
    
    # --- REFINED PROMPTS FOR BETTER, NON-REPETITIVE OUTPUT ---
    system_prompt = (
        "You are a Senior Product Strategist at a major electronics company. Your task is to provide a concise, "
        "actionable, and formal recommendation for the company's leadership based solely on the provided customer feedback. "
        "Your tone should be direct, professional, and business-focused. Do not repeat the customer feedback verbatim "
        "in your answer; instead, synthesize it into clear strategic advice. Use headings like 'Actionable Recommendation:' and 'Key Actions:'"
    )
    
    user_prompt = ""
    if pos_count > neg_count * 1.5 and pos_count > 5: # Significantly more positive reviews
        user_prompt = (
            f"Analyze the feedback for the product '{model_name}'. It has received overwhelmingly positive reviews "
            f"(Total Positive: {pos_count}). Customers frequently praise these aspects: '{pos_sample}'. "
            f"What is your primary recommendation to capitalize on this success and further enhance market position?"
        )
    elif neg_count > pos_count * 1.5 and neg_count > 5: # Significantly more negative reviews
        user_prompt = (
            f"Analyze the feedback for the product '{model_name}'. It is receiving significant negative feedback "
            f"(Total Negative: {neg_count}). The major complaints are about: '{neg_sample}'. "
            f"What is the most critical action the company must take immediately to address these issues and mitigate damage?"
        )
    else: # Mixed or neutral feedback
        user_prompt = (
            f"Analyze the mixed feedback for the product '{model_name}'. "
            f"Positive points (Total Positive: {pos_count}) include: '{pos_sample}'. "
            f"However, critical issues (Total Negative: {neg_count}) include: '{neg_sample}'. "
            f"Provide a balanced recommendation that addresses the key weaknesses while leveraging the strengths."
        )

    try:
        chat_completion = client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            model=MODEL_NAME,
            temperature=0.7, # A bit higher temperature for more creative suggestions
            max_tokens=300,  # Increased tokens for longer, more comprehensive suggestions
            stop=None # Ensure it doesn't stop prematurely
        )
        suggestion = chat_completion.choices[0].message.content.strip()
        
        # Save to cache before returning
        cache[cache_key] = suggestion
        save_cache(cache)
        
        return suggestion
    except Exception as e:
        print(f"Error during Groq API call for {model_name}: {e}")
        return f"Could not generate suggestion due to an API error: {e}"