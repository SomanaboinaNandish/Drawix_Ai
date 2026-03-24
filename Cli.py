#!/usr/bin/env python3
"""
Empathy Engine CLI
Usage: python cli.py "Your text here"
       python cli.py --interactive
"""

import argparse
import sys
import os

# Add parent dir to path so we can import app modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app import detect_emotion, compute_voice_params, build_ssml, synthesize_speech

EMOTION_ICONS = {
    "joy":        "😄",
    "anger":      "😠",
    "sadness":    "😢",
    "fear":       "😨",
    "surprise":   "😲",
    "disgust":    "🤢",
    "neutral":    "😐",
    "inquisitive":"🤔",
    "concerned":  "😟",
    "sarcastic":  "😏",
}

BAR_CHARS = "█" * 20


def render_bar(value, min_val=0.5, max_val=2.0):
    pct = (value - min_val) / (max_val - min_val)
    filled = int(pct * 20)
    return "█" * filled + "░" * (20 - filled)


def process_text(text: str, play: bool = False, output_file: str = None):
    print("\n" + "═" * 60)
    print("  🎙  EMPATHY ENGINE")
    print("═" * 60)
    print(f"\n  Input: \"{text[:80]}{'...' if len(text) > 80 else ''}\"")

    print("\n  ⏳ Analyzing emotion...")
    emotion_result = detect_emotion(text)

    icon = EMOTION_ICONS.get(emotion_result["emotion"], "💬")
    print(f"\n  {icon}  Detected Emotion:  {emotion_result['label'].upper()}")
    print(f"     Confidence:   {emotion_result['score']:.1%}")
    print(f"     Intensity:    {emotion_result['intensity']:.1%}  [{render_bar(emotion_result['intensity'], 0, 1)}]")
    print(f"     Valence:      {emotion_result['valence']:+.2f}  {'▲ Positive' if emotion_result['valence'] > 0.1 else '▼ Negative' if emotion_result['valence'] < -0.1 else '— Neutral'}")
    print(f"     Method:       {emotion_result['method']}")

    voice_params = compute_voice_params(emotion_result)
    print("\n  🔊 Voice Parameter Mapping:")
    print(f"     Rate:    {voice_params['rate']:.2f}x  [{render_bar(voice_params['rate'])}]")
    print(f"     Pitch:   {voice_params['pitch']:.2f}x  [{render_bar(voice_params['pitch'])}]")
    print(f"     Volume:  {voice_params['volume']:.2f}x  [{render_bar(voice_params['volume'])}]")
    print(f"     Emphasis: {voice_params['emphasis']}")

    print("\n  📝 Generated SSML:")
    ssml = build_ssml(text, voice_params, emotion_result)
    for line in ssml.strip().split("\n"):
        print(f"     {line}")

    print("\n  🎵 Synthesizing audio...")
    try:
        audio_path = synthesize_speech(text, voice_params, emotion_result)
        print(f"\n  ✅ Audio saved: {audio_path}")

        if output_file:
            import shutil
            shutil.copy(audio_path, output_file)
            print(f"  📁 Copied to:   {output_file}")

        if play:
            _play_audio(audio_path)

    except RuntimeError as e:
        print(f"\n  ❌ Synthesis failed: {e}")
        print("  💡 Install gTTS: pip install gTTS pydub")
        print("     or pyttsx3:   pip install pyttsx3")

    print("\n" + "═" * 60 + "\n")


def _play_audio(path: str):
    """Attempt to play audio on the system."""
    import subprocess
    import platform
    system = platform.system()
    try:
        if system == "Darwin":
            subprocess.run(["afplay", path], check=True)
        elif system == "Linux":
            subprocess.run(["aplay", path], check=True)
        elif system == "Windows":
            import winsound
            winsound.PlaySound(path, winsound.SND_FILENAME)
        print("  🔈 Audio played.")
    except Exception as e:
        print(f"  ⚠️  Playback unavailable: {e}. Open the file manually.")


def demo_mode():
    """Run a set of demo sentences covering all emotions."""
    demos = [
        ("joy",        "This is absolutely incredible! I just got promoted and I couldn't be happier!"),
        ("anger",      "This is completely UNACCEPTABLE! I've been waiting three hours and nobody helped me!"),
        ("sadness",    "I'm really sorry to hear about your loss. It must be incredibly difficult right now."),
        ("fear",       "I'm not sure we're ready for this. The risks are real and I'm genuinely worried."),
        ("surprise",   "Wait — you're telling me this happened overnight? I can't believe it!"),
        ("neutral",    "The meeting is scheduled for Tuesday at three in the afternoon."),
        ("inquisitive","Why does this keep happening? I've been wondering if there's a pattern here."),
        ("concerned",  "I'm concerned about the timeline. We really need to address this carefully."),
    ]
    print("\n🎭 EMPATHY ENGINE — DEMO MODE")
    print("Running all emotion categories...\n")
    for emotion, text in demos:
        print(f"[{emotion.upper()}]")
        process_text(text)
        print()


def main():
    parser = argparse.ArgumentParser(
        description="Empathy Engine: Emotion-aware Text-to-Speech",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python cli.py "I'm so happy today!"
  python cli.py "This is terrible!" --play
  python cli.py --interactive
  python cli.py --demo
        """
    )
    parser.add_argument("text", nargs="?", help="Text to synthesize")
    parser.add_argument("--play", action="store_true", help="Play audio after synthesis")
    parser.add_argument("--output", "-o", help="Copy audio to this path")
    parser.add_argument("--interactive", "-i", action="store_true", help="Interactive loop")
    parser.add_argument("--demo", action="store_true", help="Run all emotion demos")
    parser.add_argument("--analyze-only", action="store_true", help="Detect emotion without audio")

    args = parser.parse_args()

    if args.demo:
        demo_mode()
        return

    if args.interactive:
        print("🎙  Empathy Engine — Interactive Mode (Ctrl+C to quit)\n")
        while True:
            try:
                text = input("Enter text: ").strip()
                if not text:
                    continue
                if args.analyze_only:
                    er = detect_emotion(text)
                    vp = compute_voice_params(er)
                    print(f"  Emotion: {er['label']} (intensity: {er['intensity']:.1%})")
                    print(f"  Voice:   rate={vp['rate']:.2f}x  pitch={vp['pitch']:.2f}x  volume={vp['volume']:.2f}x\n")
                else:
                    process_text(text, play=args.play, output_file=args.output)
            except KeyboardInterrupt:
                print("\n\nGoodbye! 👋")
                break
        return

    if args.text:
        process_text(args.text, play=args.play, output_file=args.output)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()