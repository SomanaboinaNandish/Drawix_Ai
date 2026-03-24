# 🎙 Empathy Engine — Emotion-Aware Text-to-Speech

## 🚀 Overview

Empathy Engine is an AI-powered text-to-speech system that detects human emotions from text and generates expressive speech by dynamically adjusting voice parameters like pitch, rate, and volume.

## 🧠 Features

* Emotion Detection (Transformer + VADER fallback)
* Emotion → Voice Mapping (dynamic scaling)
* SSML-based expressive speech
* Multi-engine TTS (ElevenLabs, pyttsx3, gTTS)
* Web UI + CLI support

## 🏗 Tech Stack

* Python, Flask
* NLP (Transformers, VADER)
* TTS (pyttsx3, gTTS, ElevenLabs)
* HTML, CSS, JavaScript

## 📦 Installation

```bash
git clone https://github.com/your-username/empathy-engine.git
cd empathy-engine
pip install -r requirements.txt
```

## ▶️ Run Application

```bash
python app.py
```

Open: http://localhost:5000

## 🖥 CLI Usage

```bash
python cli.py "I am very happy today!"
```

## 🔄 Workflow

1. Input text
2. Emotion detection
3. Voice parameter mapping
4. SSML generation
5. Speech synthesis

## 🎯 Example

Input:
"I'm so happy today!"

Output:

* Emotion: Joy 😄
* Dynamic expressive audio

## 🌟 Future Improvements

* Real-time voice streaming
* Multilingual support
* Emotion fine-tuning

## 👨‍💻 Author

Nandish — AI Engineer Enthusiast
