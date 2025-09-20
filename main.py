import os
import sys
import json
import time
import asyncio
from datetime import datetime
from typing import List, Dict, Optional
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import requests
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
import warnings
warnings.filterwarnings("ignore")

from fastapi import FastAPI, Form, Request, HTTPException
from fastapi.responses import PlainTextResponse
import uvicorn
from twilio.twiml.messaging_response import MessagingResponse
from twilio.rest import Client
import logging

# ----------------------------
# Configuration
# ----------------------------
DEFAULT_DATA_PATH = r"C:\TonyStark\medquad (2).csv"
INDEX_DIR = "./medquad_index_cli"
REMINDERS_FILE = "./reminders.json"
USER_PROFILES_FILE = "./user_profiles.json"
USER_SESSIONS_FILE = "./user_sessions.json"

# Twilio Configuration (set these environment variables)
TWILIO_ACCOUNT_SID = os.getenv('AC0f32bae9ef0d5232cd54009921ef3e0a')
TWILIO_AUTH_TOKEN = os.getenv('fa4a62ab3369f8b967b49ed8052edafc')
TWILIO_PHONE_NUMBER = os.getenv('+14155238886')  

SUMMARIZER_MODEL = "sshleifer/distilbart-cnn-12-6"
MARIAN_PAIR = {
    "hi": ("Helsinki-NLP/opus-mt-en-hi", "Helsinki-NLP/opus-mt-hi-en"),
    "bn": ("Helsinki-NLP/opus-mt-en-bn", "Helsinki-NLP/opus-mt-bn-en"),
    "ta": ("Helsinki-NLP/opus-mt-en-ta", "Helsinki-NLP/opus-mt-ta-en"),
    "ml": ("Helsinki-NLP/opus-mt-en-mt", None),
}

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables for ML models
vectorizer = None
X = None
df = None
summarizer = None
translation_models = {}

app = FastAPI(title="WhatsApp Medical Chatbot", version="1.0.0")

# ----------------------------
# User Session Management
# ----------------------------
class UserSessionManager:
    def __init__(self):
        self.sessions = {}
        self.load_sessions()
    
    def load_sessions(self):
        try:
            if os.path.exists(USER_SESSIONS_FILE):
                with open(USER_SESSIONS_FILE, 'r') as f:
                    self.sessions = json.load(f)
        except Exception as e:
            logger.error(f"Error loading sessions: {e}")
            self.sessions = {}
    
    def save_sessions(self):
        try:
            with open(USER_SESSIONS_FILE, 'w') as f:
                json.dump(self.sessions, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving sessions: {e}")
    
    def get_session(self, phone_number: str) -> Dict:
        if phone_number not in self.sessions:
            self.sessions[phone_number] = {
                "state": "idle",
                "profile": {"age": None, "gender": None, "region": None, "lifestyle": None, "lang": "en"},
                "last_retrieved": [],
                "profile_setup_step": 0,
                "created_at": datetime.now().isoformat()
            }
            self.save_sessions()
        return self.sessions[phone_number]
    
    def update_session(self, phone_number: str, updates: Dict):
        session = self.get_session(phone_number)
        session.update(updates)
        self.save_sessions()

session_manager = UserSessionManager()

# ----------------------------
# Data Loading and ML Setup
# ----------------------------
def load_dataset(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found at {path}")
    df = pd.read_csv(path)
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    if 'question' not in df.columns or 'answer' not in df.columns:
        raise ValueError("CSV must contain 'question' and 'answer' columns")
    df = df[['question', 'answer']].fillna('')
    return df

def build_or_load_index(df: pd.DataFrame):
    texts = (df['question'].astype(str) + " ||| " + df['answer'].astype(str)).tolist()
    logger.info("Building TF-IDF index...")
    vectorizer = TfidfVectorizer(max_features=20000, ngram_range=(1,2), stop_words='english')
    X = vectorizer.fit_transform(texts)
    logger.info(f"Index built. Rows: {len(texts)}")
    return vectorizer, X, df

def get_summarizer():
    logger.info(f"Loading summarizer model: {SUMMARIZER_MODEL}")
    summarizer = pipeline("summarization", model=SUMMARIZER_MODEL, truncation=True)
    return summarizer

def load_translation_models(preferred_lang: str):
    lang = preferred_lang.lower() if preferred_lang else "en"
    if lang == "en":
        return (lambda t: t), (lambda t: t)
    
    if lang in translation_models:
        return translation_models[lang]
    
    pair = MARIAN_PAIR.get(lang)
    if pair and pair[0]:
        try:
            logger.info(f"Loading MarianMT model for en -> {lang}: {pair[0]}")
            tok_en_lang = AutoTokenizer.from_pretrained(pair[0])
            model_en_lang = AutoModelForSeq2SeqLM.from_pretrained(pair[0])
            
            def en_to_lang(text):
                inputs = tok_en_lang(text, return_tensors="pt", truncation=True, max_length=512)
                out = model_en_lang.generate(**inputs, max_length=512)
                return tok_en_lang.decode(out[0], skip_special_tokens=True)
            
            if pair[1]:
                logger.info(f"Loading MarianMT model for {lang} -> en: {pair[1]}")
                tok_lang_en = AutoTokenizer.from_pretrained(pair[1])
                model_lang_en = AutoModelForSeq2SeqLM.from_pretrained(pair[1])
                
                def lang_to_en(text):
                    inputs = tok_lang_en(text, return_tensors="pt", truncation=True, max_length=512)
                    out = model_lang_en.generate(**inputs, max_length=512)
                    return tok_lang_en.decode(out[0], skip_special_tokens=True)
            else:
                lang_to_en = lambda t: t
            
            translation_models[lang] = (en_to_lang, lang_to_en)
            return en_to_lang, lang_to_en
            
        except Exception as e:
            logger.error(f"MarianMT load failed: {str(e)}")
    
    # Fallback: identity functions
    translation_models[lang] = (lambda t: t, lambda t: t)
    return (lambda t: t), (lambda t: t)

# ----------------------------
# Core Functions
# ----------------------------
def retrieve(query: str, topk=5):
    qv = vectorizer.transform([query])
    sims = cosine_similarity(qv, X).flatten()
    idx = np.argsort(sims)[::-1][:topk]
    results = []
    for i in idx:
        results.append({
            "question": df.iloc[i]['question'],
            "answer": df.iloc[i]['answer'],
            "score": float(sims[i])
        })
    return results

def summarize_retrieved(user_query: str, retrieved: List[Dict], profile: Dict, max_length=120):
    context_texts = []
    for i, item in enumerate(retrieved, start=1):
        context_texts.append(f"Source {i} Q: {item['question']}\nA: {item['answer']}\n")
    context = "\n\n".join(context_texts)

    age = profile.get("age")
    if age is not None and age >= 60:
        tone = "formal, clear, step-by-step"
    elif age is not None and age < 18:
        tone = "simple, friendly, short sentences"
    else:
        tone = "casual, clear"

    region_hint = ""
    region = profile.get("region")
    if region:
        region_hint = f"Add one line of local prevention advice relevant to {region} if applicable."

    prompt = (
        f"You are a concise, layman-friendly public health assistant.\n"
        f"User question: {user_query}\n"
        f"Personalization: age={age}, gender={profile.get('gender')}, lifestyle={profile.get('lifestyle')}\n"
        f"Tone: {tone}. {region_hint}\n\n"
        f"Use ONLY the information from the retrieved sources below to answer. Produce:\n"
        f"1) A 2-3 sentence direct answer in plain language.\n"
        f"2) One actionable sentence if user should seek medical attention (if relevant).\n\n"
        f"Context:\n{context}\n\nAnswer:"
    )

    inputs = prompt
    if len(inputs.split()) > 1500:
        inputs = " ".join(inputs.split()[-1200:])
    
    try:
        res = summarizer(inputs, max_length=max_length, min_length=30, do_sample=False)
        summary = res[0]['summary_text'].strip()
    except Exception as e:
        logger.error(f"Summarization error: {e}")
        summary = " ".join([(item['answer'][:300] + "...") for item in retrieved[:3]])
    
    if age is not None and age >= 60:
        summary += " (Advice: seniors should monitor symptoms closely and consult a doctor early.)"
    
    return summary

def generate_quiz_from_retrieved(retrieved: List[Dict]):
    quiz = []
    if not retrieved:
        return quiz
    
    from collections import Counter
    import re
    
    corpus = " ".join([r['answer'] for r in retrieved])
    sents = re.split(r'(?<=[.!?])\s+', corpus)
    facts = [s.strip() for s in sents if 6 < len(s.split()) < 30]
    facts = facts[:6]
    
    for i, fact in enumerate(facts[:2]):
        tokens = fact.split()
        if len(tokens) < 4:
            continue
        
        correct = " ".join(tokens[-2:])
        wrong1 = " ".join(tokens[-3:-1]) if len(tokens) >= 5 else "unknown"
        wrong2 = tokens[0] if tokens else "unknown"
        
        q_text = f"Q{i+1}: Which is most related to: \"{fact[:80]}...\"?"
        options = [correct, wrong1, wrong2]
        np.random.shuffle(options)
        
        quiz.append({
            "question": q_text,
            "options": options,
            "answer": correct
        })
    
    return quiz

def find_nearby_hospitals(location_query: str, limit=5):
    try:
        if "," in location_query:
            lat, lon = [float(x.strip()) for x in location_query.split(",")[:2]]
            overpass_url = "http://overpass-api.de/api/interpreter"
            radius = 5000
            query = f"""
            [out:json][timeout:25];
            (
              node["amenity"="hospital"](around:{radius},{lat},{lon});
              way["amenity"="hospital"](around:{radius},{lat},{lon});
              relation["amenity"="hospital"](around:{radius},{lat},{lon});
            );
            out center {limit};
            """
            resp = requests.post(overpass_url, data=query, timeout=30)
            j = resp.json()
            places = []
            for el in j.get('elements', [])[:limit]:
                name = el.get('tags', {}).get('name', 'Unnamed Hospital')
                if 'lat' in el:
                    plat = el['lat']; plon = el['lon']
                else:
                    plat = el.get('center', {}).get('lat'); plon = el.get('center', {}).get('lon')
                places.append({"name": name, "lat": plat, "lon": plon, "tags": el.get('tags', {})})
            return places
    except Exception as e:
        logger.error(f"Hospital search error: {e}")
    
    # Fallback to Nominatim
    nom_url = "https://nominatim.openstreetmap.org/search"
    params = {"q": f"hospital near {location_query}", "format": "json", "limit": limit}
    try:
        resp = requests.get(nom_url, params=params, headers={"User-Agent": "WhatsApp-MedBot/1.0"})
        data = resp.json()
        places = []
        for item in data:
            places.append({
                "name": item.get('display_name'),
                "lat": item.get('lat'),
                "lon": item.get('lon')
            })
        return places
    except Exception as e:
        logger.error(f"Nominatim search error: {e}")
        return []

def load_reminders():
    if not os.path.exists(REMINDERS_FILE):
        return []
    try:
        with open(REMINDERS_FILE, "r") as f:
            return json.load(f)
    except Exception:
        return []

def save_reminder(rem):
    rems = load_reminders()
    rems.append(rem)
    with open(REMINDERS_FILE, "w") as f:
        json.dump(rems, f, indent=2)

# ----------------------------
# Message Processing
# ----------------------------
def process_message(phone_number: str, message: str) -> str:
    session = session_manager.get_session(phone_number)
    message_lower = message.lower().strip()
    
    # Handle commands
    if message_lower in ['help', 'menu', 'commands']:
        return get_help_message()
    
    if message_lower in ['setprofile', 'profile', 'setup']:
        session_manager.update_session(phone_number, {
            "state": "profile_setup",
            "profile_setup_step": 1
        })
        return "Let's set up your profile! Please enter your age (or type 'skip'):"
    
    # Handle profile setup
    if session["state"] == "profile_setup":
        return handle_profile_setup(phone_number, message)
    
    if message_lower.startswith('ask '):
        question = message[4:].strip()
        return handle_health_question(phone_number, question)
    
    if message_lower == 'quiz':
        return handle_quiz_request(phone_number)
    
    if message_lower.startswith('sos '):
        location = message[4:].strip()
        return handle_sos_request(location)
    
    if message_lower.startswith('remind '):
        reminder_text = message[7:].strip()
        return handle_reminder(reminder_text)
    
    if message_lower == 'reminders':
        return handle_show_reminders()
    
    # Default: treat as health question
    return handle_health_question(phone_number, message)

def get_help_message():
    return """🏥 *WhatsApp Medical Assistant*

*Commands:*
• `ask [question]` - Ask a health question
• `setprofile` - Set up your profile (age, gender, region, etc.)
• `quiz` - Get a quiz after asking a question
• `sos [location]` - Find nearby hospitals
• `remind [text]` - Set a health reminder
• `reminders` - View your reminders
• `help` - Show this menu

*Examples:*
• ask What are symptoms of fever?
• sos Mumbai
• remind Take medicine at 8 PM daily

Just type your health question directly, or use commands above!"""

def handle_profile_setup(phone_number: str, message: str) -> str:
    session = session_manager.get_session(phone_number)
    step = session["profile_setup_step"]
    profile = session["profile"]
    
    if step == 1:  # Age
        if message.lower() != 'skip':
            try:
                profile["age"] = int(message)
            except ValueError:
                return "Please enter a valid age number or 'skip':"
        
        session_manager.update_session(phone_number, {
            "profile": profile,
            "profile_setup_step": 2
        })
        return "Gender (M/F/Other or 'skip'):"
    
    elif step == 2:  # Gender
        if message.lower() != 'skip':
            profile["gender"] = message
        
        session_manager.update_session(phone_number, {
            "profile": profile,
            "profile_setup_step": 3
        })
        return "Your region/state (e.g., Kerala, Maharashtra or 'skip'):"
    
    elif step == 3:  # Region
        if message.lower() != 'skip':
            profile["region"] = message
        
        session_manager.update_session(phone_number, {
            "profile": profile,
            "profile_setup_step": 4
        })
        return "Lifestyle notes (e.g., smoker, diabetic, active or 'skip'):"
    
    elif step == 4:  # Lifestyle
        if message.lower() != 'skip':
            profile["lifestyle"] = message
        
        session_manager.update_session(phone_number, {
            "profile": profile,
            "profile_setup_step": 5
        })
        return "Preferred language (en/hi/bn/ta or 'skip'):"
    
    elif step == 5:  # Language
        if message.lower() != 'skip':
            profile["lang"] = message
        
        session_manager.update_session(phone_number, {
            "profile": profile,
            "state": "idle",
            "profile_setup_step": 0
        })
        
        profile_summary = f"✅ *Profile Set!*\n"
        profile_summary += f"Age: {profile.get('age', 'Not set')}\n"
        profile_summary += f"Gender: {profile.get('gender', 'Not set')}\n"
        profile_summary += f"Region: {profile.get('region', 'Not set')}\n"
        profile_summary += f"Lifestyle: {profile.get('lifestyle', 'Not set')}\n"
        profile_summary += f"Language: {profile.get('lang', 'en')}\n\n"
        profile_summary += "Now you can ask health questions!"
        
        return profile_summary

def handle_health_question(phone_number: str, question: str) -> str:
    if not question.strip():
        return "Please ask a health question. Example: 'What are symptoms of fever?'"
    
    session = session_manager.get_session(phone_number)
    profile = session["profile"]
    user_lang = profile.get('lang', 'en')
    
    try:
        # Translate question to English if needed
        if user_lang != "en":
            en_to_lang, lang_to_en = load_translation_models(user_lang)
            try:
                q_en = lang_to_en(question)
            except Exception:
                q_en = question
        else:
            q_en = question
            en_to_lang, lang_to_en = (lambda t: t), (lambda t: t)
        
        # Retrieve relevant information
        retrieved = retrieve(q_en, topk=5)
        
        # Generate summary
        summary_en = summarize_retrieved(q_en, retrieved, profile)
        
        # Translate back if needed
        if user_lang != "en":
            try:
                summary_user = en_to_lang(summary_en)
            except Exception:
                summary_user = summary_en
        else:
            summary_user = summary_en
        
        # Store retrieved results for potential quiz
        session_manager.update_session(phone_number, {
            "last_retrieved": retrieved
        })
        
        response = f"🏥 *Health Information*\n\n{summary_user}\n\n"
        response += "💡 Type 'quiz' for a quick test on this topic!"
        
        return response
        
    except Exception as e:
        logger.error(f"Error processing health question: {e}")
        return "I'm having trouble processing your question right now. Please try again later."

def handle_quiz_request(phone_number: str) -> str:
    session = session_manager.get_session(phone_number)
    retrieved = session.get("last_retrieved", [])
    
    if not retrieved:
        return "No recent health question found. Please ask a health question first!"
    
    quiz = generate_quiz_from_retrieved(retrieved)
    if not quiz:
        return "Couldn't generate a quiz from the recent topic. Try asking another question!"
    
    response = "🧠 *Quick Health Quiz*\n\n"
    for i, q in enumerate(quiz, 1):
        response += f"*{q['question']}*\n"
        for j, opt in enumerate(q['options'], 1):
            response += f"{j}. {opt}\n"
        response += f"\n*Answer: {q['answer']}*\n\n"
    
    return response

def handle_sos_request(location: str) -> str:
    if not location.strip():
        return "Please provide your location. Example: 'sos Mumbai' or 'sos 19.0760,72.8777'"
    
    try:
        hospitals = find_nearby_hospitals(location, limit=5)
        if not hospitals:
            return f"🚨 No hospitals found near '{location}'. Try a different location or contact emergency services: 108"
        
        response = f"🚨 *Nearby Hospitals - {location}*\n\n"
        for i, hospital in enumerate(hospitals, 1):
            name = hospital.get('name', 'Unnamed Hospital')
            lat = hospital.get('lat', 'N/A')
            lon = hospital.get('lon', 'N/A')
            response += f"{i}. *{name}*\n"
            if lat != 'N/A' and lon != 'N/A':
                response += f"   📍 https://maps.google.com/?q={lat},{lon}\n\n"
            else:
                response += f"   📍 Location: {name}\n\n"
        
        response += "⚠️ *Emergency: Call 108 for ambulance*"
        return response
        
    except Exception as e:
        logger.error(f"Error in SOS request: {e}")
        return "🚨 Error finding hospitals. For emergencies, call 108 immediately!"

def handle_reminder(reminder_text: str) -> str:
    if not reminder_text.strip():
        return "Please provide reminder text. Example: 'remind Take medicine at 8 PM'"
    
    try:
        reminder = {
            "text": reminder_text,
            "created_at": datetime.now().isoformat(),
            "user": "whatsapp_user"  # Could be enhanced to use phone numbers
        }
        save_reminder(reminder)
        return f"✅ *Reminder Saved!*\n\n📝 {reminder_text}\n⏰ Created: {datetime.now().strftime('%Y-%m-%d %H:%M')}"
        
    except Exception as e:
        logger.error(f"Error saving reminder: {e}")
        return "Error saving reminder. Please try again."

def handle_show_reminders() -> str:
    try:
        reminders = load_reminders()
        if not reminders:
            return "📋 No reminders found. Use 'remind [text]' to add one!"
        
        response = "📋 *Your Health Reminders*\n\n"
        for i, rem in enumerate(reminders[-10:], 1):  # Show last 10
            response += f"{i}. {rem['text']}\n"
            response += f"   ⏰ {rem.get('created_at', 'Unknown time')}\n\n"
        
        return response
        
    except Exception as e:
        logger.error(f"Error loading reminders: {e}")
        return "Error loading reminders. Please try again."

# ----------------------------
# FastAPI Endpoints
# ----------------------------
@app.get("/")
async def root():
    return {"message": "WhatsApp Medical Chatbot is running!"}


# Replace your existing webhook endpoint with this fixed version:

@app.post("/webhook", response_class=PlainTextResponse)
async def webhook(request: Request, Body: str = Form(...), From: str = Form(...)):
    """Handle incoming WhatsApp messages"""
    try:
        phone_number = From.replace("whatsapp:", "")
        message = Body.strip()
        
        logger.info(f"Received message from {phone_number}: {message}")
        
        # Process the message
        response_text = process_message(phone_number, message)
        
        logger.info(f"Sending response to {phone_number}: {response_text[:100]}...")
        
        # Return just the message content, not TwiML XML
        return response_text
        
    except Exception as e:
        logger.error(f"Error processing webhook: {e}")
        return "Sorry, I'm having technical difficulties. Please try again later."
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "models_loaded": {
            "vectorizer": vectorizer is not None,
            "summarizer": summarizer is not None,
            "dataset": df is not None
        }
    }

# ----------------------------
# Startup Events
# ----------------------------
@app.on_event("startup")
async def startup_event():
    """Initialize ML models on startup"""
    global vectorizer, X, df, summarizer
    
    try:
        logger.info("Loading dataset...")
        df = load_dataset(DEFAULT_DATA_PATH)
        
        logger.info("Building index...")
        vectorizer, X, df = build_or_load_index(df)
        
        logger.info("Loading summarizer...")
        summarizer = get_summarizer()
        
        logger.info("🚀 WhatsApp Medical Chatbot startup complete!")
        
    except Exception as e:
        logger.error(f"Startup error: {e}")
        sys.exit(1)

# ----------------------------
# Main
# ----------------------------
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--data", default=DEFAULT_DATA_PATH, help="Path to MedQuAD CSV")
    args = parser.parse_args()
    
    # Update data path if provided
    if args.data != DEFAULT_DATA_PATH:
        DEFAULT_DATA_PATH = args.data
    
    # Check Twilio configuration
    if not all([TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN, TWILIO_PHONE_NUMBER]):
        logger.warning("Twilio configuration not complete. Set TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN, TWILIO_PHONE_NUMBER")
    
    uvicorn.run(app, host=args.host, port=args.port)
