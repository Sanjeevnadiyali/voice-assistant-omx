import streamlit as st
import speech_recognition as sr
from gtts import gTTS
import os
import base64
import json
import re
from langdetect import detect, DetectorFactory
from concurrent.futures import ThreadPoolExecutor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from datetime import datetime

# ---
# CONFIGURATION
# ---
DetectorFactory.seed = 0
FAQ_FILE = "faqs.json"
SIM_THRESHOLD = 0.5

# ---
# HELPER FUNCTIONS
# ---

def clean_text(text: str) -> str:
    return re.sub(r"[^\w\s]", "", text.lower().strip())

def load_faqs():
    """Load FAQs from faqs.json with Hindi and English mappings"""
    try:
        with open(FAQ_FILE, "r", encoding="utf-8") as f:
            raw_data = json.load(f)

        faqs = {'english': {}, 'hindi': {}}
        for key, value in raw_data.items():
            hindi_q, english_q = key.split("|")
            faqs["english"][english_q.strip()] = {
                "answer": value["answer_en"],
                "keywords": value.get("keywords", [])
            }
            faqs["hindi"][hindi_q.strip()] = {
                "answer": value["answer_hi"],
                "keywords": value.get("keywords", [])
            }
        return faqs
    except Exception as e:
        st.error(f"Error loading FAQs: {e}")
        return {"english": {}, "hindi": {}}

def detect_language(text: str) -> str:
    """Detects input language: hi → Hindi, en → English"""
    try:
        lang = detect(text)
        return 'hi' if lang in ['hi', 'bn', 'pa', 'mr'] else 'en'
    except:
        return 'en'

# ---
# SPEECH
# ---

@st.cache_resource
def get_recognizer():
    r = sr.Recognizer()
    r.energy_threshold = 4000
    r.dynamic_energy_threshold = True
    return r

def recognize_speech():
    """Capture speech and detect both English & Hindi"""
    r = get_recognizer()
    try:
        with sr.Microphone() as source:
            st.info("🔍 Speak now (English or Hindi)...")
            r.adjust_for_ambient_noise(source, duration=0.8)
            audio = r.listen(source, timeout=6, phrase_time_limit=6)

        with ThreadPoolExecutor() as executor:
            en_future = executor.submit(r.recognize_google, audio, language='en-IN')
            hi_future = executor.submit(r.recognize_google, audio, language='hi-IN')

        try:
            text_en = en_future.result(timeout=3)
            if text_en:
                return text_en, 'en'
        except:
            pass

        try:
            text_hi = hi_future.result(timeout=3)
            if text_hi:
                return text_hi, 'hi'
        except:
            pass

        return None, None
    except Exception as e:
        st.warning(f"Error recognizing speech: {e}")
        return None, None

def text_to_speech(text, lang='en'):
    try:
        tts = gTTS(text=text, lang='hi' if lang == 'hi' else 'en', slow=False)
        filename = f"response_{datetime.now().timestamp()}.mp3"
        tts.save(filename)
        return filename
    except Exception as e:
        st.error(f"TTS Error: {e}")
        return None

def play_audio(file_path):
    if file_path and os.path.exists(file_path):
        with open(file_path, "rb") as f:
            audio_bytes = f.read()
        audio_base64 = base64.b64encode(audio_bytes).decode("utf-8")
        st.markdown(
            f'<audio autoplay><source src="data:audio/mp3;base64,{audio_base64}"></audio>',
            unsafe_allow_html=True,
        )
        os.remove(file_path)

# ---
# FAQ ENGINE
# ---

class FAQEngine:
    def __init__(self):
        self.faqs = load_faqs()
        self.vectorizer = {
            "english": TfidfVectorizer(ngram_range=(1, 2)),
            "hindi": TfidfVectorizer(ngram_range=(1, 2))
        }
        self.matrix = {"english": None, "hindi": None}
        self.entries = {"english": [], "hindi": []}
        self.answers = {"english": [], "hindi": []}
        self._train()

    def _train(self):
        for lc in ["english", "hindi"]:
            self.entries[lc] = []
            self.answers[lc] = []
            for q, data in self.faqs[lc].items():
                self.entries[lc].append(clean_text(q))
                self.answers[lc].append(data["answer"])
                for kw in data.get("keywords", []):
                    self.entries[lc].append(clean_text(kw))
                    self.answers[lc].append(data["answer"])
            if self.entries[lc]:
                self.matrix[lc] = self.vectorizer[lc].fit_transform(self.entries[lc])

    def _best_match(self, user_text, lang_code):
        if self.matrix[lang_code] is None or self.matrix[lang_code].shape[0] == 0:
            return None, 0.0
        vec = self.vectorizer[lang_code].transform([clean_text(user_text)])
        sims = cosine_similarity(vec, self.matrix[lang_code]).flatten()
        best_idx = int(np.argmax(sims))
        best_score = float(sims[best_idx])
        if best_score >= SIM_THRESHOLD:
            return self.answers[lang_code][best_idx], best_score
        else:
            return None, best_score

    def answer(self, user_text, lang_code):
        ans, score = self._best_match(user_text, "hindi" if lang_code == "hi" else "english")
        return ans

# ---
# UI
# ---

def setup_ui():
    st.set_page_config(page_title="OMX AI Assistant", layout="wide")
    st.markdown("""
    <style>
    .stApp { background: #e6f2ff; color: #003366; }
    .title { text-align: center; font-size: 2.5em; margin-bottom: 0.5em; color: #003366; }
    .stButton>button { background: #004080; color: white; border-radius: 20px; font-weight: bold; }
    </style>
    """, unsafe_allow_html=True)

def display_conversation(user_q, lang, answer):
    with st.expander("Conversation", expanded=True):
        st.markdown(f"<b>You:</b> {user_q} ({'Hindi' if lang == 'hi' else 'English'})", unsafe_allow_html=True)
        if answer:
            st.markdown(f"<b>Assistant:</b> {answer} ({'Hindi' if lang == 'hi' else 'English'})", unsafe_allow_html=True)
            audio_file = text_to_speech(answer, lang)
            play_audio(audio_file)
        else:
            fallback = {
                'hi': "माफ़ करें, मैं इस प्रश्न का उत्तर नहीं जानता।",
                'en': "Sorry, I don't know the answer to this question."
            }.get(lang, "Sorry, I don't know.")
            st.warning(fallback)
            audio_file = text_to_speech(fallback, lang)
            play_audio(audio_file)

# --- MAIN APP ---

def main():
    setup_ui()
    engine = FAQEngine()

    st.markdown("<h1 class='title'>OMX Digital AI Assistant</h1>", unsafe_allow_html=True)
    st.markdown("<center><i>Ask questions in English or Hindi</i></center>", unsafe_allow_html=True)

    tab1, tab2 = st.tabs(['🎤 Voice Query', '✏️ Text Query'])
    with tab1:
        if st.button("Start Recording"):
            with st.spinner("Listening..."):
                question, lang = recognize_speech()
            if question:
                ans = engine.answer(question, lang)
                display_conversation(question, lang, ans)
            else:
                st.warning("Could not understand audio. Please try again.")
    with tab2:
        question = st.text_input("Type your question (English or Hindi):")
        if st.button("Submit"):
            if question.strip():
                lang = detect_language(question)
                ans = engine.answer(question, lang)
                display_conversation(question, lang, ans)

if __name__ == "__main__":
    main()