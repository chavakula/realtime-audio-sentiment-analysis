#!/usr/bin/env python3
"""
Generate sample call audio files using macOS TTS (``say`` command).

Produces three test WAV files in ``sample_calls/``:
  - positive_call.wav   – happy customer
  - negative_call.wav   – frustrated customer
  - escalating_call.wav – starts neutral, becomes negative

Requires: macOS + ffmpeg (``brew install ffmpeg``)
"""

import os
import subprocess
import sys
import tempfile

SAMPLE_DIR = os.path.join(os.path.dirname(__file__), "sample_calls")

CONVERSATIONS: dict[str, list[tuple[str, str, str]]] = {
    # (role, voice, text)
    "positive_call": [
        ("caller", "Samantha", "Hi, I recently bought a laptop from you and I absolutely love it."),
        ("agent", "Daniel", "That's wonderful to hear! I'm glad you're enjoying your new laptop."),
        ("caller", "Samantha", "The performance is amazing and the battery lasts all day. I'm very impressed."),
        ("agent", "Daniel", "Thank you so much for sharing that. Is there anything else I can help you with today?"),
        ("caller", "Samantha", "No, everything is perfect. You've been really helpful. Thank you!"),
        ("agent", "Daniel", "You're welcome! Have a great day."),
    ],
    "negative_call": [
        ("caller", "Samantha", "I've been waiting on hold for thirty minutes. This is completely unacceptable."),
        ("agent", "Daniel", "I sincerely apologize for the long wait. Let me help you right away."),
        ("caller", "Samantha", "My order arrived damaged for the second time. I am extremely frustrated."),
        ("agent", "Daniel", "I understand your frustration and I'm very sorry about that. Let me fix this for you."),
        ("caller", "Samantha", "I want a full refund. This is the worst service I have ever experienced."),
        ("agent", "Daniel", "I completely understand. I will process your refund immediately."),
        ("caller", "Samantha", "I'm seriously considering never buying from you again."),
    ],
    "escalating_call": [
        ("caller", "Samantha", "Hello, I have a question about my monthly bill."),
        ("agent", "Daniel", "Of course! I'd be happy to help you with your billing question."),
        ("caller", "Samantha", "There's a charge here I don't recognize. Could you explain it?"),
        ("agent", "Daniel", "Let me pull up your account and take a look at that charge."),
        ("caller", "Samantha", "I see I was charged twice for the same service. That's a bit frustrating."),
        ("agent", "Daniel", "I do see the duplicate charge. I apologize for that error."),
        ("caller", "Samantha", "This happened last month too and nobody fixed it. I'm getting really angry."),
        ("agent", "Daniel", "I'm very sorry this is a recurring problem. Let me escalate this."),
        ("caller", "Samantha", "I want to speak with a manager right now. This is absolutely unacceptable!"),
    ],
}


def generate_sample(name: str, script: list[tuple[str, str, str]]) -> str:
    """Generate a single WAV file from a script of (role, voice, text) tuples."""
    os.makedirs(SAMPLE_DIR, exist_ok=True)
    output_path = os.path.join(SAMPLE_DIR, f"{name}.wav")

    temp_files: list[str] = []

    try:
        for i, (role, voice, text) in enumerate(script):
            # Generate AIFF via macOS say
            aiff_path = os.path.join(tempfile.gettempdir(), f"_{name}_{i}.aiff")
            wav_path = os.path.join(tempfile.gettempdir(), f"_{name}_{i}.wav")
            temp_files.extend([aiff_path, wav_path])

            subprocess.run(
                ["say", "-v", voice, "-o", aiff_path, text],
                check=True,
                capture_output=True,
            )
            # Convert AIFF → 16kHz mono WAV
            subprocess.run(
                [
                    "ffmpeg", "-y",
                    "-i", aiff_path,
                    "-ar", "16000",
                    "-ac", "1",
                    wav_path,
                ],
                check=True,
                capture_output=True,
            )

        # Build ffmpeg concat filter to join all segments with 0.5s silence
        silence_path = os.path.join(tempfile.gettempdir(), f"_{name}_silence.wav")
        temp_files.append(silence_path)
        subprocess.run(
            [
                "ffmpeg", "-y",
                "-f", "lavfi",
                "-i", "anullsrc=r=16000:cl=mono",
                "-t", "0.5",
                silence_path,
            ],
            check=True,
            capture_output=True,
        )

        # Build concat list
        concat_list_path = os.path.join(tempfile.gettempdir(), f"_{name}_list.txt")
        temp_files.append(concat_list_path)
        with open(concat_list_path, "w") as f:
            for i in range(len(script)):
                wav_path = os.path.join(tempfile.gettempdir(), f"_{name}_{i}.wav")
                f.write(f"file '{wav_path}'\n")
                if i < len(script) - 1:
                    f.write(f"file '{silence_path}'\n")

        subprocess.run(
            [
                "ffmpeg", "-y",
                "-f", "concat",
                "-safe", "0",
                "-i", concat_list_path,
                "-ar", "16000",
                "-ac", "1",
                output_path,
            ],
            check=True,
            capture_output=True,
        )
        print(f"  ✓ {output_path}")
        return output_path

    finally:
        for f in temp_files:
            if os.path.exists(f):
                os.remove(f)


def main() -> None:
    # Check prerequisites
    for cmd in ["say", "ffmpeg"]:
        if subprocess.run(["which", cmd], capture_output=True).returncode != 0:
            print(f"Error: '{cmd}' not found. ", file=sys.stderr)
            if cmd == "ffmpeg":
                print("Install with: brew install ffmpeg", file=sys.stderr)
            sys.exit(1)

    print(f"Generating sample audio files in '{SAMPLE_DIR}/'...\n")

    for name, script in CONVERSATIONS.items():
        generate_sample(name, script)

    print(f"\n✅ Done! {len(CONVERSATIONS)} files generated in '{SAMPLE_DIR}/'")
    print("Use these to test the real-time sentiment analysis.")


if __name__ == "__main__":
    main()
