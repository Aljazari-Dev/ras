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
#   BANNER_PATH=/absolute/path/to/aljazari_banner.png
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
#     -F "banner=@/path/to/aljazari_banner.png" -o with_banner.png
#
#   # Ride (Space call) with uploaded person + uploaded banner:
#   curl -X POST http://localhost:$PORT/ride \
#     -F "person=@/path/to/person.jpg" \
#     -F "with_banner=1" \
#     -F "banner=@/path/to/aljazari_banner.png" -o ride_output.png
# -----------------------------------------------------------

import io
import os
import tempfile
from pathlib import Path

from dotenv import load_dotenv
from flask import Flask, jsonify, request, send_file
from PIL import Image

# ---------------- IP REGISTRY STATE ----------------
latest_pi_ip = None
latest_time = None
latest_cruzr_ip = None
latest_cruzr_time = None

# ---------------- IMAGE/SPACE CONFIG ----------------
SPACE_ID = "akhaliq/Qwen-Image-Edit-2509"

# App directory + default assets path (portable across Linux/Windows)
APP_DIR = Path(__file__).resolve().parent
DEFAULT_BANNER = APP_DIR / "assets" / "aljazari_banner.png"

# You can override via env: BANNER_PATH=/abs/path/to/aljazari_banner.png
BANNER_PATH = Path(os.getenv("BANNER_PATH", str(DEFAULT_BANNER)))

DEFAULT_PROMPT = (
    " a person riding the dragon, seated astride, hands holding a horn or spine, "
    "realistic contact shadows, correct perspective, high detail "
    "Negative prompt: extra limbs, deformed, blurry, low quality, distortion"
)
DEFAULT_NEGATIVE = "text, logo, watermark, extra limbs, blur, artifacts, man in front of dragon face"
SEED = 0
TRUE_CFG_SCALE = 1.0
NUM_STEPS = 20
GUIDANCE_SCALE = 1.0

# ---------------- APP INIT ----------------
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")
PORT = int(os.getenv("PORT", "5001"))
HOST = os.getenv("HOST", "0.0.0.0")
DEBUG = os.getenv("DEBUG", "1") in ("1", "true", "True")

app = Flask(__name__)

# Lazy import for gradio_client so IP endpoints work even without image deps
_gradio_ready = False
_client = None
_handle_file = None
_AppError = Exception

def _init_gradio():
    """Initialize gradio client once (lazy)."""
    global _gradio_ready, _client, _handle_file, _AppError
    if _gradio_ready:
        return
    try:
        from gradio_client import Client, handle_file
        from gradio_client.exceptions import AppError
        _AppError = AppError
        _handle_file = handle_file
        # Init client
        _client = Client(SPACE_ID, hf_token=HF_TOKEN)
        _gradio_ready = True
    except Exception:
        _gradio_ready = False

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

# ---------------- IMAGE EDIT: RIDE (Space call) ----------------
@app.route("/ride", methods=["POST"])
def ride():
    """
    multipart/form-data:
      - person (file)  [required]
      - prompt (str)   [optional]
      - negative (str) [optional]
      - steps (int)    [optional]
      - guidance (float) [optional]
      - seed (int)     [optional]
      - with_banner (0/1) [optional]
      - banner (file)  [optional]  <- upload banner per request
      - banner_height_px (int) [optional]
      - banner_ratio (float)   [optional; ignored if height given]
    returns: image/png
    """
    if "person" not in request.files:
        return jsonify({"error": "missing file field 'person'"}), 400

    # Initialize gradio client
    _init_gradio()
    if not _gradio_ready or _client is None or _handle_file is None:
        return jsonify({
            "error": "space_unavailable",
            "detail": "gradio_client not initialized. Check dependencies and HF_TOKEN."
        }), 503

    person_file = request.files["person"]
    if not person_file.filename:
        return jsonify({"error": "empty filename"}), 400

    prompt   = request.form.get("prompt", DEFAULT_PROMPT)
    negative = request.form.get("negative", DEFAULT_NEGATIVE)
    try:
        steps    = int(request.form.get("steps", NUM_STEPS))
        guidance = float(request.form.get("guidance", GUIDANCE_SCALE))
        seed     = int(request.form.get("seed", SEED))
    except ValueError:
        return jsonify({"error": "bad_params", "detail": "steps/guidance/seed must be numeric"}), 400

    with_banner = request.form.get("with_banner", "0") in ("1", "true", "True")
    banner_height_px = request.form.get("banner_height_px")
    banner_height_px = int(banner_height_px) if banner_height_px else None
    banner_ratio = request.form.get("banner_ratio")
    banner_ratio = float(banner_ratio) if banner_ratio else 0.18

    # You must set DRAGON_PATH to a valid file (we ship it via env or keep alongside the server)
    # To avoid cross-OS confusion, let users set DRAGON_PATH via env. If not set, fail gracefully.
    DRAGON_PATH = os.getenv("DRAGON_PATH")
    if not DRAGON_PATH or not os.path.exists(DRAGON_PATH):
        return jsonify({
            "error": "dragon_not_found",
            "detail": f"Set DRAGON_PATH to a valid image file. Current: {DRAGON_PATH}"
        }), 500

    tmp_dir = tempfile.mkdtemp(prefix="ride_")
    tmp_person_path = os.path.join(tmp_dir, person_file.filename)
    person_file.save(tmp_person_path)

    try:
        # call Space with two images
        job = _client.submit(
            _handle_file(tmp_person_path),   # image1: person
            _handle_file(DRAGON_PATH),       # image2: dragon
            prompt, seed, TRUE_CFG_SCALE, negative, steps, guidance,
            api_name="/edit_images"
        )
        result_path = job.result()

        # open result
        img = Image.open(result_path).convert("RGB")

        # optional banner (from upload or from disk)
        if with_banner:
            banner_file = request.files.get("banner")
            if banner_file:
                banner_img = Image.open(banner_file.stream)
            else:
                if not BANNER_PATH.exists():
                    return jsonify({"error": "banner_not_found", "detail": str(BANNER_PATH)}), 500
                banner_img = Image.open(BANNER_PATH)
            img = attach_banner(img, banner_img, banner_height_px=banner_height_px, banner_ratio=banner_ratio)

        buf = io.BytesIO()
        img.save(buf, format="PNG")
        buf.seek(0)
        return send_file(buf, mimetype="image/png", download_name="ride_output.png")
    except _AppError as e:
        return jsonify({"error": "space_error", "detail": str(e)}), 503
    except Exception as e:
        return jsonify({"error": "server_error", "detail": str(e)}), 500
    finally:
        # Clean tmp files
        try:
            if os.path.exists(tmp_person_path):
                os.remove(tmp_person_path)
            if os.path.isdir(tmp_dir):
                os.rmdir(tmp_dir)
        except Exception:
            pass

# ---------------- IMAGE: ADD BANNER ONLY ----------------
@app.route("/add_banner", methods=["POST"])
def add_banner_endpoint():
    """
    Append the banner below any uploaded image.
    multipart/form-data:
      - image (file)           [required]
      - banner (file)          [optional]  <- upload banner per request
      - banner_height_px (int) [optional]
      - banner_ratio (float)   [optional] default 0.18
    returns: image/png
    """
    if "image" not in request.files:
        return jsonify({"error": "missing file field 'image'"}), 400

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

    base_img = Image.open(image_file.stream).convert("RGB")

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

# ---------------- DEBUG HELPERS ----------------
@app.route("/debug/env", methods=["GET"])
def debug_env():
    # Minimal leak-free status (no secrets)
    return jsonify({
        "PORT": PORT,
        "HOST": HOST,
        "DEBUG": DEBUG,
        "SPACE_ID": SPACE_ID,
        "BANNER_PATH": str(BANNER_PATH),
        "BANNER_EXISTS": BANNER_PATH.exists(),
        "DRAGON_PATH": os.getenv("DRAGON_PATH"),
        "DRAGON_EXISTS": os.path.exists(os.getenv("DRAGON_PATH", "")) if os.getenv("DRAGON_PATH") else False,
        "HF_TOKEN_SET": bool(HF_TOKEN),
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
