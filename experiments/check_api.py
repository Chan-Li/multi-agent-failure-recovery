"""Sanity-check the OpenAI-compatible API: list models and send one tiny request.

Usage: python experiments/check_api.py [--model NAME]
"""
from __future__ import annotations

import argparse
import os
import time

from dotenv import load_dotenv
from openai import OpenAI


def main():
    load_dotenv()
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="api-glm-5.3-flash")
    args = ap.parse_args()

    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"],
                    base_url=os.environ.get("OPENAI_BASE_URL"))
    models = [m.id for m in client.models.list().data]
    print("available models:")
    for m in models:
        print("  -", m)

    model = args.model
    if model:
        t0 = time.time()
        r = client.chat.completions.create(
            model=model, max_tokens=20,
            messages=[{"role": "user", "content": "Reply with the single word: ok"}])
        print(f"\n[{model}] {r.choices[0].message.content!r}  "
              f"({time.time() - t0:.2f}s, usage={r.usage})")


if __name__ == "__main__":
    main()
