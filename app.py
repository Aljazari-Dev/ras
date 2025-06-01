from flask import Flask, request, jsonify

app = Flask(__name__)
latest_pi_ip = None
latest_time = None

# NEW: Store Cruzr IP
latest_cruzr_ip = None
latest_cruzr_time = None

@app.route('/register_pi_ip', methods=['POST'])
def register_pi_ip():
    global latest_pi_ip, latest_time
    data = request.get_json()
    latest_pi_ip = data.get('ip')
    latest_time = data.get('time')
    return jsonify({"status": "ok"})

@app.route('/get_pi_ip', methods=['GET'])
def get_pi_ip():
    if latest_pi_ip:
        return jsonify({"ip": latest_pi_ip, "time": latest_time})
    else:
        return jsonify({"error": "No IP registered yet"}), 404

# === NEW ENDPOINTS FOR CRUZR ===
@app.route('/register_cruzr_ip', methods=['POST'])
def register_cruzr_ip():
    global latest_cruzr_ip, latest_cruzr_time
    data = request.get_json()
    latest_cruzr_ip = data.get('ip')
    latest_cruzr_time = data.get('time')
    return jsonify({"status": "ok"})

@app.route('/get_cruzr_ip', methods=['GET'])
def get_cruzr_ip():
    if latest_cruzr_ip:
        return jsonify({"ip": latest_cruzr_ip, "time": latest_cruzr_time})
    else:
        return jsonify({"error": "No Cruzr IP registered yet"}), 404

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
