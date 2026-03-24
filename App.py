

import os
import uuid
import json
import math
from flask import Flask, request, jsonify, send_file, render_template, send_from_directory

app = Flask(__name__)

ELEVENLABS_API_KEY = "ef726772e1cabd27b1003e313332488887d492200a57e08735150bb907c8c962"

AUDIO_OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "audio_output")
os.makedirs(AUDIO_OUTPUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Emotion Detection
# ---------------------------------------------------------------------------

def detect_emotion(text: str) -> dict:
    """
    Detect emotion from text using a layered approach:
    1. Try transformers (j-hartmann/emotion-english-distilroberta-base) for 7-class emotion
    2. Fall back to VADER sentiment + keyword rules for granular emotions
    Returns: { emotion, label, score, intensity, valence }
    """
    try:
        from transformers import pipeline
        classifier = pipeline(
            "text-classification",
            model="j-hartmann/emotion-english-distilroberta-base",
            top_k=None,
            device=-1
        )
        results = classifier(text[:512])[0]
        results_sorted = sorted(results, key=lambda x: x["score"], reverse=True)
        top = results_sorted[0]
        emotion = top["label"].lower()
        score = top["score"]

        valence_map = {
            "joy": 1.0, "surprise": 0.3, "neutral": 0.0,
            "sadness": -0.6, "anger": -0.8, "disgust": -0.7, "fear": -0.5
        }
        valence = valence_map.get(emotion, 0.0)
        intensity = _compute_intensity(text, score)

        return {
            "emotion": emotion,
            "label": _friendly_label(emotion),
            "score": round(score, 3),
            "intensity": round(intensity, 3),
            "valence": round(valence, 3),
            "method": "transformer",
            "all_scores": {r["label"].lower(): round(r["score"], 3) for r in results_sorted}
        }
    except Exception:
        pass

    # VADER fallback
    return _vader_emotion(text)


def _vader_emotion(text: str) -> dict:
    """VADER + keyword heuristics for granular emotion detection."""
    try:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        analyzer = SentimentIntensityAnalyzer()
        scores = analyzer.polarity_scores(text)
        compound = scores["compound"]
    except ImportError:
        # Pure keyword fallback
        compound = _keyword_sentiment(text)
        scores = {"compound": compound, "pos": max(0, compound), "neg": max(0, -compound), "neu": 0.5}

    text_lower = text.lower()

    # Granular emotion via keywords
    emotion_keywords = {
        "joy":      ["amazing", "wonderful", "fantastic", "love", "excellent", "great", "happy", "excited", "thrilled", "delighted", "awesome", "brilliant"],
        "anger":    ["angry", "furious", "outraged", "hate", "terrible", "disgusting", "unacceptable", "ridiculous", "absurd", "awful"],
        "sadness":  ["sad", "sorry", "unfortunate", "miss", "lost", "crying", "heartbroken", "disappointed", "grief", "mourn"],
        "fear":     ["afraid", "scared", "worried", "anxious", "nervous", "terrified", "concerning", "risk", "danger", "threat"],
        "surprise": ["wow", "unbelievable", "incredible", "shocking", "unexpected", "sudden", "really", "wait", "what"],
        "disgust":  ["disgusting", "gross", "revolting", "horrible", "nauseating", "appalling", "vile"],
        "inquisitive": ["why", "how", "what", "when", "where", "wondering", "curious", "question", "thinking", "pondering", "?"],
        "concerned":["concerned", "worry", "careful", "please", "need", "must", "important", "urgent", "serious"],
        "sarcastic":["obviously", "clearly", "sure", "totally", "right", "thanks a lot", "great job", "wonderful job"],
    }

    hit_counts = {}
    for emo, words in emotion_keywords.items():
        hits = sum(1 for w in words if w in text_lower)
        if hits:
            hit_counts[emo] = hits

    if hit_counts:
        best_kw = max(hit_counts, key=hit_counts.get)
    else:
        best_kw = None

    # Combine VADER + keywords
    if compound >= 0.5:
        emotion = best_kw if best_kw in ("joy", "surprise") else "joy"
    elif compound >= 0.1:
        emotion = best_kw if best_kw else "neutral"
    elif compound <= -0.5:
        emotion = best_kw if best_kw in ("anger", "sadness", "fear", "disgust") else "anger"
    elif compound <= -0.1:
        emotion = best_kw if best_kw else "sadness"
    else:
        emotion = best_kw if best_kw else "neutral"

    valence_map = {
        "joy": 1.0, "surprise": 0.3, "neutral": 0.0, "inquisitive": 0.1,
        "sadness": -0.6, "anger": -0.8, "disgust": -0.7, "fear": -0.5,
        "concerned": -0.3, "sarcastic": -0.2
    }

    intensity = _compute_intensity(text, abs(compound))
    return {
        "emotion": emotion,
        "label": _friendly_label(emotion),
        "score": round(abs(compound), 3),
        "intensity": round(intensity, 3),
        "valence": round(valence_map.get(emotion, 0.0), 3),
        "method": "vader",
        "all_scores": hit_counts
    }


def _keyword_sentiment(text: str) -> float:
    pos = ["good", "great", "happy", "love", "excellent", "wonderful", "nice", "best"]
    neg = ["bad", "hate", "terrible", "worst", "awful", "horrible", "poor"]
    t = text.lower()
    score = sum(1 for w in pos if w in t) - sum(1 for w in neg if w in t)
    return max(-1.0, min(1.0, score * 0.3))


def _compute_intensity(text: str, base_score: float) -> float:
    """
    Intensity [0..1] amplified by:
    - exclamation marks
    - all-caps words
    - intensifier words
    - text length (longer = more deliberate)
    """
    exclamations = text.count("!") * 0.1
    caps_words = sum(1 for w in text.split() if w.isupper() and len(w) > 1) * 0.08
    intensifiers = sum(1 for w in ["very", "extremely", "so", "really", "absolutely",
                                    "completely", "totally", "utterly", "incredibly"]
                       if w in text.lower()) * 0.07
    intensity = base_score + exclamations + caps_words + intensifiers
    return min(1.0, intensity)


def _friendly_label(emotion: str) -> str:
    labels = {
        "joy": "Joyful / Happy",
        "anger": "Angry / Frustrated",
        "sadness": "Sad / Disappointed",
        "fear": "Fearful / Anxious",
        "surprise": "Surprised",
        "disgust": "Disgusted",
        "neutral": "Neutral / Calm",
        "inquisitive": "Inquisitive / Curious",
        "concerned": "Concerned / Worried",
        "sarcastic": "Sarcastic",
    }
    return labels.get(emotion, emotion.capitalize())


# ---------------------------------------------------------------------------
# Voice Parameter Mapping
# ---------------------------------------------------------------------------

# Base profiles for each emotion
EMOTION_PROFILES = {
    "joy": {
        "rate_base": 1.20,   # faster, energetic
        "pitch_base": 1.15,  # higher pitch
        "volume_base": 1.10, # slightly louder
        "emphasis": "strong",
        "pause_factor": 0.8,
    },
    "anger": {
        "rate_base": 1.15,   # faster, clipped
        "pitch_base": 0.95,  # slightly lower, tense
        "volume_base": 1.15, # louder
        "emphasis": "strong",
        "pause_factor": 0.7,
    },
    "sadness": {
        "rate_base": 0.80,   # slower, heavy
        "pitch_base": 0.90,  # lower pitch
        "volume_base": 0.85, # quieter
        "emphasis": "none",
        "pause_factor": 1.4,
    },
    "fear": {
        "rate_base": 1.10,   # slightly faster, nervous
        "pitch_base": 1.10,  # higher, strained
        "volume_base": 0.90, # slightly quieter
        "emphasis": "moderate",
        "pause_factor": 0.9,
    },
    "surprise": {
        "rate_base": 1.05,
        "pitch_base": 1.20,  # notably higher on surprise
        "volume_base": 1.05,
        "emphasis": "strong",
        "pause_factor": 0.85,
    },
    "disgust": {
        "rate_base": 0.92,   # slower, deliberate
        "pitch_base": 0.88,
        "volume_base": 0.95,
        "emphasis": "moderate",
        "pause_factor": 1.1,
    },
    "neutral": {
        "rate_base": 1.00,
        "pitch_base": 1.00,
        "volume_base": 1.00,
        "emphasis": "none",
        "pause_factor": 1.0,
    },
    "inquisitive": {
        "rate_base": 0.95,   # thoughtful pace
        "pitch_base": 1.08,  # rising intonation
        "volume_base": 0.95,
        "emphasis": "moderate",
        "pause_factor": 1.1,
    },
    "concerned": {
        "rate_base": 0.88,
        "pitch_base": 0.95,
        "volume_base": 0.90,
        "emphasis": "moderate",
        "pause_factor": 1.2,
    },
    "sarcastic": {
        "rate_base": 0.90,   # slow, deliberate
        "pitch_base": 1.05,
        "volume_base": 1.00,
        "emphasis": "moderate",
        "pause_factor": 1.15,
    },
}


def compute_voice_params(emotion_result: dict) -> dict:
    """
    Scale base profile by intensity for expressive modulation.
    intensity=0  → neutral params
    intensity=1  → fully emotional params
    """
    emotion = emotion_result["emotion"]
    intensity = emotion_result["intensity"]
    profile = EMOTION_PROFILES.get(emotion, EMOTION_PROFILES["neutral"])
    neutral = EMOTION_PROFILES["neutral"]

    def interp(base, neutral_val):
        return neutral_val + (base - neutral_val) * intensity

    rate   = interp(profile["rate_base"],   neutral["rate_base"])
    pitch  = interp(profile["pitch_base"],  neutral["pitch_base"])
    volume = interp(profile["volume_base"], neutral["volume_base"])

    # Clamp to safe ranges
    rate   = round(max(0.5, min(2.0, rate)),   3)
    pitch  = round(max(0.5, min(2.0, pitch)),  3)
    volume = round(max(0.5, min(1.5, volume)), 3)

    return {
        "rate": rate,
        "pitch": pitch,
        "volume": volume,
        "emphasis": profile["emphasis"],
        "pause_factor": round(interp(profile["pause_factor"], 1.0), 3),
    }


# ---------------------------------------------------------------------------
# SSML Generation
# ---------------------------------------------------------------------------

def build_ssml(text: str, voice_params: dict, emotion_result: dict) -> str:
    """
    Build SSML with prosody tags for emotion-aware TTS.
    Maps pitch/rate/volume to SSML percentage offsets.
    """
    rate_pct   = int((voice_params["rate"]   - 1.0) * 100)
    pitch_pct  = int((voice_params["pitch"]  - 1.0) * 100)
    volume_db  = round((voice_params["volume"] - 1.0) * 6, 1)  # rough dB

    rate_str   = f"{rate_pct:+d}%"
    pitch_str  = f"{pitch_pct:+d}%"
    volume_str = f"{volume_db:+.1f}dB"

    # Add strategic pauses based on punctuation
    processed = text
    pause_ms = int(300 * voice_params["pause_factor"])
    processed = processed.replace(". ", f'. <break time="{pause_ms}ms"/> ')
    processed = processed.replace("? ", f'? <break time="{int(pause_ms * 0.8)}ms"/> ')
    processed = processed.replace("! ", f'! <break time="{int(pause_ms * 0.7)}ms"/> ')

    # Emphasis on strong-emotion short words
    if voice_params["emphasis"] == "strong" and len(text.split()) <= 20:
        import re
        # capitalize important words emphasis
        processed = re.sub(
            r'\b([A-Z]{2,})\b',
            r'<emphasis level="strong">\1</emphasis>',
            processed
        )

    ssml = f"""<speak>
  <prosody rate="{rate_str}" pitch="{pitch_str}" volume="{volume_str}">
    {processed}
  </prosody>
</speak>"""
    return ssml


# ---------------------------------------------------------------------------
# TTS Synthesis
# ---------------------------------------------------------------------------


def _try_elevenlabs(text: str, voice_params: dict, emotion_result: dict, output_path: str) -> bool:
    try:
        from elevenlabs import VoiceSettings
        from elevenlabs.client import ElevenLabs

        client = ElevenLabs(api_key=ELEVENLABS_API_KEY)

        emotion = emotion_result["emotion"]
        intensity = emotion_result["intensity"]

        # 🔥 Map emotion → ElevenLabs style
        stability = max(0.2, 1.0 - intensity)   # less stable = more expressive
        similarity_boost = 0.7

        style_map = {
            "joy": 0.8,
            "anger": 0.9,
            "sadness": 0.3,
            "fear": 0.6,
            "surprise": 0.9,
            "neutral": 0.2
        }

        style = style_map.get(emotion, 0.5)

        audio = client.text_to_speech.convert(
            text=text,
            voice_id="EXAVITQu4vr4xnSDxMaL",  # default Rachel voice
            model_id="eleven_multilingual_v2",
            voice_settings=VoiceSettings(
                stability=stability,
                similarity_boost=similarity_boost,
                style=style,
                use_speaker_boost=True
            )
        )

        with open(output_path, "wb") as f:
            for chunk in audio:
                f.write(chunk)

        print(f"[ElevenLabs] Used | emotion={emotion} | intensity={intensity}")
        return True

    except Exception as e:
        print(f"[ElevenLabs] Failed: {e}")
        return False


def synthesize_speech(text: str, voice_params: dict, emotion_result: dict) -> str:
    """
    Priority:
    1. ElevenLabs (best quality, limited)
    2. pyttsx3 (offline fallback)
    3. gTTS fallback
    """

    output_id = str(uuid.uuid4())[:8]
    emotion = emotion_result["emotion"]

    # --- Try ElevenLabs first ---
    mp3_path = os.path.join(AUDIO_OUTPUT_DIR, f"empathy_{emotion}_{output_id}_el.mp3")
    if ELEVENLABS_API_KEY and _try_elevenlabs(text, voice_params, emotion_result, mp3_path):
        return mp3_path

    # --- Fallback to pyttsx3 ---
    wav_path = os.path.join(AUDIO_OUTPUT_DIR, f"empathy_{emotion}_{output_id}.wav")
    if _try_pyttsx3(text, voice_params, wav_path):
        return wav_path

    # --- Final fallback gTTS ---
    mp3_path = os.path.join(AUDIO_OUTPUT_DIR, f"empathy_{emotion}_{output_id}.mp3")
    if _try_gtts_modulated(text, voice_params, mp3_path):
        return mp3_path

    raise RuntimeError("All TTS engines failed.")

def _try_pyttsx3(text: str, voice_params: dict, output_path: str) -> bool:
    """
    pyttsx3 with real vocal parameter control.
    On Windows, SAPI5 natively supports rate, volume, and voice selection.

    Rate:   default ~200 wpm. We scale it by our rate multiplier.
            Joy(1.2x) → ~240 wpm | Sadness(0.8x) → ~160 wpm
    Volume: 0.0–1.0 scale.
    Pitch:  pyttsx3 doesn't expose pitch directly, but we compensate
            by adding subtle rate variation to reinforce the emotional feel.
    """
    try:
        import pyttsx3
        engine = pyttsx3.init()

        # --- Rate (speed) ---
        # Default is usually 200 wpm; scale by emotion multiplier
        default_rate = engine.getProperty("rate") or 200
        new_rate = int(default_rate * voice_params["rate"])
        # Keep within safe speech bounds
        new_rate = max(80, min(400, new_rate))
        engine.setProperty("rate", new_rate)

        # --- Volume ---
        # voice_params volume is centered at 1.0; SAPI5 wants 0.0–1.0
        new_volume = min(1.0, max(0.1, voice_params["volume"]))
        engine.setProperty("volume", new_volume)

        # --- Voice selection for pitch simulation ---
        # SAPI5 on Windows typically has male + female voices.
        # We pick female voice for high-pitch emotions (joy, surprise, fear)
        # and male voice for low-pitch emotions (anger, sadness, disgust).
        voices = engine.getProperty("voices")
        if voices and len(voices) > 1:
            pitch = voice_params["pitch"]
            if pitch >= 1.08:
                # High-pitch emotion → prefer female voice (index 1 on most Windows)
                female_voices = [v for v in voices if "female" in v.name.lower() or "zira" in v.name.lower() or "hazel" in v.name.lower()]
                if female_voices:
                    engine.setProperty("voice", female_voices[0].id)
                else:
                    engine.setProperty("voice", voices[1].id)
            elif pitch <= 0.93:
                # Low-pitch emotion → prefer male voice (index 0)
                male_voices = [v for v in voices if "male" in v.name.lower() or "david" in v.name.lower() or "mark" in v.name.lower()]
                if male_voices:
                    engine.setProperty("voice", male_voices[0].id)
                else:
                    engine.setProperty("voice", voices[0].id)

        print(f"[pyttsx3] rate={new_rate} wpm | volume={new_volume:.2f} | pitch_target={voice_params['pitch']:.2f}x")

        engine.save_to_file(text, output_path)
        engine.runAndWait()

        # Give file system a moment to flush (Windows quirk)
        import time
        time.sleep(0.3)

        if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
            return True
        print("[pyttsx3] Output file empty or missing.")
        return False

    except Exception as e:
        print(f"[pyttsx3] Error: {e}")
        return False


def _try_gtts_modulated(text: str, voice_params: dict, output_path: str) -> bool:
    """
    gTTS with pydub speed modulation.
    Requires ffmpeg for mp3 export; falls back to raw if not available.
    """
    try:
        from gtts import gTTS
        slow_mode = voice_params["rate"] < 0.88
        tts = gTTS(text=text, lang="en", slow=slow_mode)
        raw_path = output_path.replace(".mp3", "_raw.mp3")
        tts.save(raw_path)

        try:
            from pydub import AudioSegment
            audio = AudioSegment.from_mp3(raw_path)

            rate = voice_params["rate"]
            if abs(rate - 1.0) > 0.05:
                # Frame-rate trick: changes playback speed
                new_frame_rate = int(audio.frame_rate * rate)
                audio = audio._spawn(audio.raw_data, overrides={"frame_rate": new_frame_rate})
                audio = audio.set_frame_rate(44100)

            volume = voice_params["volume"]
            if abs(volume - 1.0) > 0.05:
                db_change = 20 * math.log10(max(0.01, volume))
                audio = audio + db_change

            audio.export(output_path, format="mp3")
            os.remove(raw_path)
            print(f"[gTTS+pydub] rate={rate:.2f}x | volume={volume:.2f}x")
        except Exception as pydub_err:
            print(f"[gTTS] pydub modulation failed ({pydub_err}), using raw audio.")
            import shutil
            shutil.move(raw_path, output_path)

        return True
    except ImportError:
        return False
    except Exception as e:
        print(f"[gTTS] Error: {e}")
        return False


# ---------------------------------------------------------------------------
# Flask Routes
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/analyze", methods=["POST"])
def analyze():
    """Analyze text emotion without generating audio."""
    data = request.get_json()
    text = (data or {}).get("text", "").strip()
    if not text:
        return jsonify({"error": "No text provided"}), 400

    emotion_result = detect_emotion(text)
    voice_params = compute_voice_params(emotion_result)
    ssml = build_ssml(text, voice_params, emotion_result)

    return jsonify({
        "emotion": emotion_result,
        "voice_params": voice_params,
        "ssml": ssml,
    })


@app.route("/synthesize", methods=["POST"])
def synthesize():
    """Full pipeline: detect emotion → compute params → generate audio."""
    data = request.get_json()
    text = (data or {}).get("text", "").strip()
    if not text:
        return jsonify({"error": "No text provided"}), 400
    if len(text) > 2000:
        return jsonify({"error": "Text too long (max 2000 chars)"}), 400

    emotion_result = detect_emotion(text)
    voice_params = compute_voice_params(emotion_result)
    ssml = build_ssml(text, voice_params, emotion_result)

    try:
        audio_path = synthesize_speech(text, voice_params, emotion_result)
        filename = os.path.basename(audio_path)
        audio_url = f"/audio/{filename}"
    except RuntimeError as e:
        return jsonify({"error": str(e)}), 500

    return jsonify({
        "emotion": emotion_result,
        "voice_params": voice_params,
        "ssml": ssml,
        "audio_url": audio_url,
        "filename": filename,
    })


@app.route("/audio/<filename>")
def serve_audio(filename):
    """Serve generated audio files (wav or mp3)."""
    safe_name = os.path.basename(filename)
    ext = safe_name.rsplit(".", 1)[-1].lower()
    mime = "audio/wav" if ext == "wav" else "audio/mpeg"
    return send_from_directory(AUDIO_OUTPUT_DIR, safe_name, mimetype=mime)


@app.route("/health")
def health():
    return jsonify({"status": "ok", "service": "Empathy Engine"})


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000) 