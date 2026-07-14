# app.py — Unified server (IP registry + image edit + banner)
# -----------------------------------------------------------
# Requirements:
#   pip install -r requirements.txt
#
# Environment (optional but recommended):
#   HF_TOKEN=hf_...               # for Hugging Face Space
#   PORT=5001                     # Render/Heroku will inject PORT automatically
#   HOST=0.0.0.0
#   DEBUG=1
#   BANNER_PATH=/absolute/path/to/1.png
#
# Run:
#   python app.py
#
# Test examples:
#   # Register Pi IP:
#   curl -X POST http://localhost:5001/register_pi_ip  \
#     -H "Content-Type: application/json" \
#     -d "{\"ip\":\"192.168.68.200\",\"time\":\"2025-10-01T09:00:00\"}"
#
#   # Get Pi IP:
#   curl http://localhost:5001/get_pi_ip
#
#   # Register Cruzr IP:
#   curl -X POST http://localhost:5001/register_cruzr_ip \
#     -H "Content-Type: application/json" \
#     -d "{\"ip\":\"192.168.68.110\",\"time\":\"2025-10-01T09:05:00\"}"
#
#   # Add banner to any image (using repo banner):
#   curl -X POST http://localhost:$PORT/add_banner \
#     -F "image=@/path/to/photo.jpg" -o with_banner.png
#
#   # Add banner to any image (uploading banner per request):
#   curl -X POST http://localhost:$PORT/add_banner \
#     -F "image=@/path/to/photo.jpg" \
#     -F "banner=@/path/to/1.png" -o with_banner.png
#
#   # Ride (Space call) with uploaded person + uploaded banner:
#   curl -X POST http://localhost:$PORT/ride \
#     -F "person=@/path/to/person.jpg" \
#     -F "with_banner=1" \
#     -F "banner=@/path/to/1.png" -o ride_output.png
# -----------------------------------------------------------

import io
import os
import sys
import tempfile
from pathlib import Path
import requests

try:
    sys.stdout.reconfigure(encoding='utf-8')
except AttributeError:
    pass

from dotenv import load_dotenv
from flask import Flask, jsonify, request, send_file
from PIL import Image
from google import genai

# ---------------- IP REGISTRY STATE ----------------
latest_pi_ip = None
latest_time = None
latest_cruzr_ip = None
latest_cruzr_time = None

# ---------------- IMAGE CONFIG ----------------
MODEL_ID = "gemini-3.1-flash-image"

# App directory + default assets path (portable across Linux/Windows)
APP_DIR = Path(__file__).resolve().parent
DEFAULT_BANNER = APP_DIR / "assets" / "1.png"

# You can override via env: BANNER_PATH=/abs/path/to/1.png
BANNER_PATH = Path(os.getenv("BANNER_PATH", str(DEFAULT_BANNER)))

DEFAULT_PROMPT = "Perform a highly realistic digital facelift and beauty enhancement. Smooth the skin, remove blemishes and wrinkles, and enhance facial symmetry and youthfulness while perfectly preserving the person's core identity, facial structure, and realism. High quality, professional photography."

# ---------------- APP INIT ----------------
load_dotenv()
PORT = int(os.getenv("PORT", "5001"))
HOST = os.getenv("HOST", "0.0.0.0")
DEBUG = os.getenv("DEBUG", "1") in ("1", "true", "True")

app = Flask(__name__)

# Lazy init for GenAI
_genai_client = None

def _init_genai():
    global _genai_client
    if _genai_client is None:
        api_key = os.getenv("Gemini_API_Key")
        if not api_key:
            return False
        _genai_client = genai.Client(api_key=api_key)
    return True

# ---------------- UTIL: BANNER ATTACH ----------------
def attach_banner(base_img: Image.Image,
                  banner_img: Image.Image,
                  banner_height_px: int | None = None,
                  banner_ratio: float | None = 0.18,
                  align: str = "center") -> Image.Image:
    """
    Append banner_img BELOW base_img.
    - If banner_height_px is provided, use it.
    - Else compute height = base_img.height * banner_ratio (default 18%).
    - Banner is resized to base width (keeping aspect); then cropped or padded to target height.
    - align: 'center' | 'left' | 'right' (horizontal placement if needed in the future).
    """
    base = base_img.convert("RGBA")
    banner = banner_img.convert("RGBA")

    W = base.width
    if banner_height_px is None:
        banner_height_px = max(1, int(base.height * (banner_ratio or 0.18)))

    # Resize banner to base width keeping aspect ratio
    scale = W / banner.width
    resized = banner.resize((W, max(1, int(banner.height * scale))), Image.LANCZOS)

    # If banner taller than target -> crop; if shorter -> pad
    if resized.height > banner_height_px:
        top = (resized.height - banner_height_px) // 2
        banner_final = resized.crop((0, top, W, top + banner_height_px))
    elif resized.height < banner_height_px:
        pad = Image.new("RGBA", (W, banner_height_px), (0, 0, 0, 0))
        y = (banner_height_px - resized.height) // 2
        x = 0
        if align == "left":
            x = 0
        elif align == "right":
            x = 0
        pad.paste(resized, (x, y), resized)
        banner_final = pad
    else:
        banner_final = resized

    out = Image.new("RGBA", (W, base.height + banner_height_px), (0, 0, 0, 0))
    out.paste(base, (0, 0), base)
    out.paste(banner_final, (0, base.height), banner_final)
    return out.convert("RGB")

# ---------------- BASIC PAGES ----------------
@app.route("/", methods=["GET"])
def index():
    return jsonify({"ok": True, "service": "ip+banner+ride"})

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})

# ---------------- IP REGISTRY ENDPOINTS ----------------
@app.route('/register_pi_ip', methods=['POST'])
def register_pi_ip():
    global latest_pi_ip, latest_time
    data = request.get_json(silent=True) or {}
    latest_pi_ip = data.get('ip')
    latest_time = data.get('time')
    return jsonify({"status": "ok", "ip": latest_pi_ip, "time": latest_time})

@app.route('/get_pi_ip', methods=['GET'])
def get_pi_ip():
    if latest_pi_ip:
        return jsonify({"ip": latest_pi_ip, "time": latest_time})
    else:
        return jsonify({"error": "No IP registered yet"}), 404

@app.route('/register_cruzr_ip', methods=['POST'])
def register_cruzr_ip():
    global latest_cruzr_ip, latest_cruzr_time
    data = request.get_json(silent=True) or {}
    latest_cruzr_ip = data.get('ip')
    latest_cruzr_time = data.get('time')
    return jsonify({"status": "ok", "ip": latest_cruzr_ip, "time": latest_cruzr_time})

@app.route('/get_cruzr_ip', methods=['GET'])
def get_cruzr_ip():
    if latest_cruzr_ip:
        return jsonify({"ip": latest_cruzr_ip, "time": latest_cruzr_time})
    else:
        return jsonify({"error": "No Cruzr IP registered yet"}), 404



# ---------------- IMAGE: FACELIFT AI + BANNER ----------------
@app.route("/add_banner", methods=["POST"])
def add_banner_endpoint():
    """
    Process image through SDXL Img2Img and append the banner below it.
    """
    if "image" not in request.files:
        return jsonify({"error": "missing file field 'image'"}), 400

    _init_genai()
    if _genai_client is None:
        return jsonify({
            "error": "api_unavailable",
            "detail": "Gemini API key not configured. Add Gemini_API_Key to .env file."
        }), 503

    image_file = request.files["image"]
    if not image_file.filename:
        return jsonify({"error": "empty filename"}), 400

    banner_height_px = request.form.get("banner_height_px")
    try:
        banner_height_px = int(banner_height_px) if banner_height_px else None
    except ValueError:
        return jsonify({"error": "bad_params", "detail": "banner_height_px must be int"}), 400

    banner_ratio = request.form.get("banner_ratio")
    try:
        banner_ratio = float(banner_ratio) if banner_ratio else 0.18
    except ValueError:
        return jsonify({"error": "bad_params", "detail": "banner_ratio must be float"}), 400

    prompt = request.form.get("prompt", DEFAULT_PROMPT)

    try:
        # Open the uploaded image
        original_img = Image.open(image_file.stream)
        # Gemini image models usually require RGB format
        if original_img.mode != "RGB":
            original_img = original_img.convert("RGB")
            
        # Call Gemini (Nano Banana) Face Enhancement / Editing
        # Pass both the image and the prompt describing the facelift effect
        response = _genai_client.models.generate_content(
            model=MODEL_ID,
            contents=[
                original_img,
                prompt
            ],
        )
        
        result_img = None
        if hasattr(response, 'parts') and response.parts:
            for part in response.parts:
                if part.inline_data and hasattr(part.inline_data, "data"):
                    # Extract bytes and load into PIL
                    img_bytes = part.inline_data.data
                    import io
                    result_img = Image.open(io.BytesIO(img_bytes))
                    break
                    
        if result_img is None:
            return jsonify({"error": "generation_failed", "detail": "No image returned by Gemini"}), 500

        base_img = result_img.convert("RGB")

        # Use uploaded banner if provided; otherwise fallback to disk path
        banner_file = request.files.get("banner")
        if banner_file:
            banner_img = Image.open(banner_file.stream)
        else:
            if not BANNER_PATH.exists():
                return jsonify({"error": "banner_not_found", "detail": str(BANNER_PATH)}), 500
            banner_img = Image.open(BANNER_PATH)

        out = attach_banner(base_img, banner_img, banner_height_px=banner_height_px, banner_ratio=banner_ratio)
        buf = io.BytesIO()
        out.save(buf, format="PNG")
        buf.seek(0)
        return send_file(buf, mimetype="image/png", download_name="with_banner.png")
    except Exception as e:
        return jsonify({"error": "server_error", "detail": str(e)}), 500

# ---------------- DEBUG HELPERS ----------------
@app.route("/debug/env", methods=["GET"])
def debug_env():
    # Minimal leak-free status (no secrets)
    return jsonify({
        "PORT": PORT,
        "HOST": HOST,
        "DEBUG": DEBUG,
        "MODEL_ID": MODEL_ID,
        "BANNER_PATH": str(BANNER_PATH),
        "BANNER_EXISTS": BANNER_PATH.exists(),
        "DRAGON_PATH": os.getenv("DRAGON_PATH"),
        "DRAGON_EXISTS": os.path.exists(os.getenv("DRAGON_PATH", "")) if os.getenv("DRAGON_PATH") else False,
    })

@app.route("/debug/files", methods=["GET"])
def debug_files():
    assets_dir = APP_DIR / "assets"
    files = []
    if assets_dir.exists():
        for p in assets_dir.iterdir():
            files.append({
                "name": p.name,
                "path": str(p),
                "is_file": p.is_file(),
                "size": p.stat().st_size if p.is_file() else None
            })
    return jsonify({
        "app_dir": str(APP_DIR),
        "assets_dir": str(assets_dir),
        "assets_list": files
    })

# ---------------- MAIN ----------------
if __name__ == "__main__":
    app.run(host=HOST, port=PORT, debug=DEBUG)
