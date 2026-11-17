"""Verify your Gemini setup before running the app.

    python check_setup.py

It checks that your API key is present, lists the Gemini models your key can use, and makes one
tiny test call with the model configured in analyzer.py (MODEL_NAME / GEMINI_MODEL).
Use it whenever you see a 404 'model not found / no longer available' error.
"""
import os
import sys

from google import genai
from google.genai import errors

from analyzer import MODEL_NAME


def main() -> int:
    key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not key:
        print("X No API key found. Set GEMINI_API_KEY in this terminal first (see README).")
        return 1
    print("OK API key found.")

    client = genai.Client(api_key=key)
    try:
        usable = []
        for model in client.models.list():
            name = getattr(model, "name", "") or ""
            actions = getattr(model, "supported_actions", None)
            if "gemini" in name and (actions is None or "generateContent" in actions):
                usable.append(name.replace("models/", ""))
    except errors.APIError as exc:
        print(f"X Could not list models ({getattr(exc, 'code', '?')}): {getattr(exc, 'message', exc)}")
        print("  Check that the key is correct and has not been deleted or restricted.")
        return 1

    print(f"\nGemini models your key can use ({len(usable)}):")
    for name in sorted(usable):
        print(f"  - {name}")

    print(f"\nTesting configured model: {MODEL_NAME}")
    try:
        reply = client.models.generate_content(model=MODEL_NAME, contents="Reply with the single word: ready")
        print(f"OK Model responded: {(reply.text or '').strip()[:60]}")
        print("\nAll good. You can run: streamlit run app.py")
        return 0
    except errors.APIError as exc:
        code = getattr(exc, "code", None)
        print(f"X Test call failed ({code}): {getattr(exc, 'message', exc)}")
        if code == 404:
            print("\n  The model is retired or unavailable to your key. Pick a current name from the list above and set it:")
            print("    Windows (Command Prompt): set GEMINI_MODEL=<model-name>")
            print("    Windows (PowerShell):     $env:GEMINI_MODEL=\"<model-name>\"")
            print("    Mac / Linux:              export GEMINI_MODEL=\"<model-name>\"")
        elif code == 429:
            print("\n  Rate limit or quota reached. Wait a minute and run this again.")
        elif code in (400, 401, 403):
            print("\n  The key was rejected. Create a fresh key at https://aistudio.google.com/apikey")
        return 1


if __name__ == "__main__":
    sys.exit(main())
