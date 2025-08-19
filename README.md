# voice-assistant-omx
Streamlit-powered multilingual voice assistant featuring speech recognition, language detection, TF-IDF vectorization, and real-time audio responses with gTTS integration.

**Prerequisites**
Python 3.8+
Microphon
Speakers/headphones

**Installation and Running**
1. Clone and Navigate to Prjoect
   git clone https://github.com/your-username/voice-assistant-omx.git
   cd voice-assistant-omx
2. Install Dependencies
   pip install streamlit speechrecognition gtts langdetect scikit-learn numpy pyaudio
3. Run the application
   streamlit run voice-assistant-omx.py
4. Open browser to http://localhost:8501

✨ **Key Features**
🎤 Voice input in English & Hindi
🗣️ Audio responses in detected language
🤖 Smart FAQ matching using AI
💬 Dual interface: Voice & Text input
⚡ Real-time processing

🎯 **Basic Usage**
Voice Tab: Click "Start Recording" and speak your question
Text Tab: Type your question and click "Submit"
Receive both text and audio responses

🛠️ **Troubleshooting**
Audio issues: Install pyaudio using pipwin install pyaudio (Windows) or sudo apt install python3-pyaudio (Linux)
Port busy: Use streamlit run voice-assistant-omx.py --server.port 8502
