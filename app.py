import os
import base64
from flask import Flask, render_template, request
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024  # 10 MB max upload

def get_client():
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY environment variable is not set.")
    return Groq(api_key=api_key)


def build_prompt(colours, pattern, size, description):
    colour_str = ", ".join(colours) if colours else "unknown"
    parts = [f"I saw a {size} bird with {colour_str} colouring."]
    if pattern:
        parts.append(f"Its pattern was: {pattern}.")
    if description:
        parts.append(f"Additional details: {description}.")
    parts.append(
        "Based on this description, suggest exactly 3 bird species this could be. "
        "For each bird, write the species name as a heading (using **Bird Name**) "
        "followed by exactly 3 sentences describing the bird and why it matches. "
        "Keep the language simple enough for a 10-year-old birdwatcher."
    )
    return " ".join(parts)


def ask_groq_text(prompt):
    response = get_client().chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a friendly expert birdwatcher who helps young children "
                    "identify birds. You only talk about birds and bird identification. "
                    "Always respond with exactly 3 bird species suggestions."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        temperature=0.4,
        max_tokens=1024,
    )
    return response.choices[0].message.content


def ask_groq_vision(prompt, image_bytes, mime_type):
    b64 = base64.standard_b64encode(image_bytes).decode("utf-8")
    data_url = f"data:{mime_type};base64,{b64}"
    response = get_client().chat.completions.create(
        model="meta-llama/llama-4-scout-17b-16e-instruct",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a friendly expert birdwatcher who helps young children "
                    "identify birds. You only talk about birds and bird identification. "
                    "Always respond with exactly 3 bird species suggestions."
                ),
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": data_url}},
                ],
            },
        ],
        temperature=0.4,
        max_tokens=1024,
    )
    return response.choices[0].message.content


def parse_birds(raw_text):
    """Split the AI response into a list of (name, description) tuples."""
    import re

    birds = []
    # Match **Bird Name** followed by text until the next **Bird Name** or end
    pattern = re.compile(r"\*\*(.+?)\*\*\s*(.*?)(?=\*\*|$)", re.DOTALL)
    matches = pattern.findall(raw_text)
    for name, body in matches:
        name = name.strip()
        body = body.strip()
        if name and body:
            birds.append({"name": name, "description": body})
    # Fallback: return raw text as a single block if parsing failed
    if not birds:
        birds = [{"name": "Result", "description": raw_text.strip()}]
    return birds[:3]


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html", birds=None, error=None)


@app.route("/identify", methods=["POST"])
def identify():
    colours = request.form.getlist("colours")
    pattern = request.form.get("pattern", "").strip()
    size = request.form.get("size", "").strip()
    description = request.form.get("description", "").strip()
    picture = request.files.get("picture")

    if not colours and not pattern and not size and not description:
        return render_template(
            "index.html",
            birds=None,
            error="Please fill in at least one field before identifying!",
        )

    prompt = build_prompt(colours, pattern, size, description)

    try:
        if picture and picture.filename:
            mime_type = picture.content_type or "image/jpeg"
            image_bytes = picture.read()
            raw = ask_groq_vision(prompt, image_bytes, mime_type)
        else:
            raw = ask_groq_text(prompt)

        birds = parse_birds(raw)
        return render_template("index.html", birds=birds, error=None)

    except Exception as e:
        return render_template(
            "index.html",
            birds=None,
            error=f"Something went wrong talking to the AI: [{type(e).__name__}] {e}",
        )


if __name__ == "__main__":
    app.run(debug=True)
