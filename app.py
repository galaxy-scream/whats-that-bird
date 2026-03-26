import os
import re
import base64
import requests as http
from flask import Flask, render_template, request
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024


def get_client():
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY environment variable is not set.")
    return Groq(api_key=api_key)


SYSTEM_PROMPT = (
    "You are a knowledgeable birdwatching expert helping young birdwatchers identify birds. "
    "You only discuss birds and bird identification. "
    "Always reply with exactly 3 bird species in the exact format requested. "
    "Only suggest species that are native to or commonly found in the specified location. "
    "Use British English (colour not color, favourite not favorite, etc.)."
)

RESPONSE_FORMAT = (
    "\n\nFor each of the 3 species, use this EXACT format with no changes:\n\n"
    "**[Common Bird Name]**\n"
    "Description: [Exactly 3 sentences: what it looks like and why it matches the description.]\n"
    "Wingspan: [e.g. 45–55 cm]\n"
    "Weight: [e.g. 80–120 g]\n"
    "Diet: [brief, e.g. Insects and berries]\n"
    "Habitat: [brief, e.g. Woodland edges and gardens]\n"
)


def build_prompt(colours, pattern, size, description, location):
    colour_str = ", ".join(colours) if colours else "unspecified"
    parts = [f"I spotted a {size} bird with {colour_str} colouring."]
    if pattern:
        parts.append(f"Pattern: {pattern}.")
    if description:
        parts.append(f"Extra details: {description}.")
    if location:
        parts.append(f"Location: {location}.")
    parts.append(
        "Please suggest 3 bird species native to this location that match this description."
        + RESPONSE_FORMAT
    )
    return " ".join(parts)


def call_groq_text(prompt):
    return get_client().chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        temperature=0.4,
        max_tokens=1400,
    ).choices[0].message.content


def call_groq_vision(prompt, image_bytes, mime_type):
    b64 = base64.standard_b64encode(image_bytes).decode("utf-8")
    data_url = f"data:{mime_type};base64,{b64}"
    return get_client().chat.completions.create(
        model="meta-llama/llama-4-scout-17b-16e-instruct",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": data_url}},
                ],
            },
        ],
        temperature=0.4,
        max_tokens=1400,
    ).choices[0].message.content


def get_bird_image(name):
    """Fetch a bird photo URL from Wikipedia."""
    try:
        slug = name.replace(" ", "_")
        resp = http.get(
            f"https://en.wikipedia.org/api/rest_v1/page/summary/{slug}",
            headers={"User-Agent": "WhatsThatBird/1.0 (educational app)"},
            timeout=4,
        )
        if resp.ok:
            return resp.json().get("thumbnail", {}).get("source", "")
    except Exception:
        pass
    return ""


def parse_birds(raw_text):
    birds = []
    blocks = re.split(r"\*\*(.+?)\*\*", raw_text)
    # blocks: ['intro', 'Name1', 'body1', 'Name2', 'body2', ...]
    i = 1
    while i + 1 < len(blocks):
        name = blocks[i].strip()
        body = blocks[i + 1]

        def field(label):
            m = re.search(rf"{label}:\s*(.+?)(?:\n|$)", body)
            return m.group(1).strip() if m else ""

        birds.append({
            "name": name,
            "description": field("Description"),
            "wingspan": field("Wingspan"),
            "weight": field("Weight"),
            "diet": field("Diet"),
            "habitat": field("Habitat"),
            "image": get_bird_image(name),
        })
        i += 2

    if not birds:
        birds = [{"name": "Unknown", "description": raw_text.strip(),
                  "image": "", "wingspan": "", "weight": "", "diet": "", "habitat": ""}]
    return birds[:3]


@app.route("/")
def index():
    return render_template("index.html", birds=None, error=None)


@app.route("/identify", methods=["POST"])
def identify():
    colours = request.form.getlist("colours")
    pattern = request.form.get("pattern", "").strip()
    size = request.form.get("size", "").strip()
    description = request.form.get("description", "").strip()
    location = request.form.get("location", "").strip()
    picture = request.files.get("picture")

    if not any([colours, pattern, size, description]):
        return render_template(
            "index.html", birds=None,
            error="Please fill in at least one field before identifying!",
        )

    prompt = build_prompt(colours, pattern, size, description, location)

    try:
        if picture and picture.filename:
            raw = call_groq_vision(prompt, picture.read(), picture.content_type or "image/jpeg")
        else:
            raw = call_groq_text(prompt)
        birds = parse_birds(raw)
        return render_template("index.html", birds=birds, error=None)

    except Exception as e:
        return render_template(
            "index.html", birds=None,
            error=f"Something went wrong: [{type(e).__name__}] {e}",
        )


@app.route("/health")
def health():
    from flask import jsonify
    return jsonify({"status": "ok", "api_key_set": bool(os.environ.get("GROQ_API_KEY"))})


@app.route("/logbook")
def logbook():
    return render_template("logbook.html")


if __name__ == "__main__":
    app.run(debug=True)
