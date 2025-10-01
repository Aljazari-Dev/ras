# app.py — Unified server (IP registry + image edit + banner)
# -----------------------------------------------------------
# Requirements:
#   pip install flask pillow gradio_client python-dotenv
# Optional:
#   set HF_TOKEN in a .env file (same folder) for Hugging Face auth
#
# Run:
#   python app.py
# Test (examples):
#   - Register Pi IP:
#       curl -X POST http://localhost:5001/register_pi_ip  -H "Content-Type: application/json" -d "{\"ip\":\"192.168.68.200\",\"time\":\"2025-10-01T09:00:00\"}"
#   - Get Pi IP:
#       curl http://localhost:5001/get_pi_ip
#   - Register Cruzr IP:
#       curl -X POST http://localhost:5001/register_cruzr_ip -H "Content-Type: application/json" -d "{\"ip\":\"192.168.68.110\",\"time\":\"2025-10-01T09:05:00\"}"
#   - Get Cruzr IP:
#       curl http://localhost:5001/get_cruzr_ip
#   - Health:
#       curl http://localhost:5001/health
#   - Ride (multipart: send a person photo file; returns PNG):
#       curl -X POST http://localhost:5001/ride \
#         -F "person=@/path/to/person.jpg" \
#         -F "with_banner=1" -o ride_output.png
#   - Add banner to any image (returns PNG):
#       curl -X POST http://localhost:5001/add_banner \
#         -F "image=@/path/to/image.jpg" -o with_banner.png
# -----------------------------------------------------------

import os, io, tempfile
from dotenv import load_dotenv
from flask import Flask, request, jsonify, send_file
from PIL import Image

# ---------------- IP REGISTRY STATE ----------------
latest_pi_ip = None
latest_time = None
latest_cruzr_ip = None
latest_cruzr_time = None

# ---------------- IMAGE/SPACE CONFIG ----------------
SPACE_ID = "akhaliq/Qwen-Image-Edit-2509"

# ⚠️ Update these paths to your actual files (Windows example shown)
DRAGON_PATH  = r"C:\Users\LENOVO\PycharmProjects\ridedragon\dragon.jpeg"
BANNER_PATH  = r"C:\Users\LENOVO\PycharmProjects\ridedragon\aljazari_banner.png"

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

# Lazy import so the IP endpoints still work even if gradio_client is missing
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
        _client = Client(SPACE_ID, hf_token=HF_TOKEN)
        _gradio_ready = True
    except Exception as e:
        # We won’t crash the server; image endpoints will return a helpful error.
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
    - align: 'center' | 'left' | 'right' (horizontal placement if ever needed).
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

# ---------------- BASIC HEALTH ----------------
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
      - banner_height_px (int) [optional]
      - banner_ratio (float)   [optional; ignored if height given]
    returns: image/png
    """
    if "person" not in request.files:
        return jsonify({"error": "missing file field 'person'"}), 400

    if not os.path.exists(DRAGON_PATH):
        return jsonify({"error": "dragon_not_found", "detail": DRAGON_PATH}), 500

    # Init gradio client if needed
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
    steps    = int(request.form.get("steps", NUM_STEPS))
    guidance = float(request.form.get("guidance", GUIDANCE_SCALE))
    seed     = int(request.form.get("seed", SEED))

    with_banner = request.form.get("with_banner", "0") in ("1", "true", "True")

    banner_height_px = request.form.get("banner_height_px")
    banner_height_px = int(banner_height_px) if banner_height_px else None
    banner_ratio = request.form.get("banner_ratio")
    banner_ratio = float(banner_ratio) if banner_ratio else 0.18

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

        # optional banner
        if with_banner:
            if not os.path.exists(BANNER_PATH):
                return jsonify({"error": "banner_not_found", "detail": BANNER_PATH}), 500
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
        try:
            if os.path.exists(tmp_person_path): os.remove(tmp_person_path)
            if os.path.isdir(tmp_dir): os.rmdir(tmp_dir)
        except Exception:
            pass

# ---------------- IMAGE: ADD BANNER ONLY ----------------
@app.route("/add_banner", methods=["POST"])
def add_banner_endpoint():
    """
    Append the banner below any uploaded image.
    multipart/form-data:
      - image (file)           [required]
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
    banner_height_px = int(banner_height_px) if banner_height_px else None
    banner_ratio = request.form.get("banner_ratio")
    banner_ratio = float(banner_ratio) if banner_ratio else 0.18

    if not os.path.exists(BANNER_PATH):
        return jsonify({"error": "banner_not_found", "detail": BANNER_PATH}), 500

    base_img = Image.open(image_file.stream).convert("RGB")
    banner_img = Image.open(BANNER_PATH)

    out = attach_banner(base_img, banner_img, banner_height_px=banner_height_px, banner_ratio=banner_ratio)
    buf = io.BytesIO()
    out.save(buf, format="PNG")
    buf.seek(0)
    return send_file(buf, mimetype="image/png", download_name="with_banner.png")

# ---------------- MAIN ----------------
if __name__ == "__main__":
    app.run(host=HOST, port=PORT, debug=DEBUG)
