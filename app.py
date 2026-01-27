from flask import Flask, render_template, request, send_file, send_from_directory
import os
import pandas as pd

from ai_detect_yolo import detect_road_damage
from ai_video_detect import analyze_video

app = Flask(__name__)

# ------------------------
# FOLDERS
# ------------------------
UPLOAD_FOLDER = "uploads"
AI_UPLOAD_FOLDER = "ai_uploads"
AI_OUTPUT_FOLDER = "ai_outputs"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(AI_UPLOAD_FOLDER, exist_ok=True)
os.makedirs(AI_OUTPUT_FOLDER, exist_ok=True)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["AI_UPLOAD_FOLDER"] = AI_UPLOAD_FOLDER
app.config["AI_OUTPUT_FOLDER"] = AI_OUTPUT_FOLDER

latest_ai_result = None

# ------------------------
# SERVE FILES
# ------------------------
@app.route("/uploads/<filename>")
def uploaded_file(filename):
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)

@app.route("/ai_uploads/<filename>")
def ai_uploaded_file(filename):
    return send_from_directory(app.config["AI_UPLOAD_FOLDER"], filename)

@app.route("/ai_outputs/<filename>")
def ai_output_file(filename):
    return send_from_directory(app.config["AI_OUTPUT_FOLDER"], filename)

# ------------------------
# HOME
# ------------------------
@app.route("/")
def home():
    return render_template("index.html")

# ------------------------
# UPLOAD PAGE
# ------------------------
@app.route("/upload")
def upload():
    return render_template("upload.html")

# ------------------------
# ANALYZE CSV + VIDEO
# ------------------------
@app.route("/analyze", methods=["POST"])
def analyze():
    csv_file = request.files.get("file")
    video_file = request.files.get("video")

    if not csv_file:
        return "No CSV file uploaded"

    # ---------- CSV ----------
    csv_path = os.path.join(app.config["UPLOAD_FOLDER"], csv_file.filename)
    csv_file.save(csv_path)

    data = pd.read_csv(csv_path)

    # ---------- PCI ----------
    data["PCI"] = 100 - (
        0.4 * data["IRI"] +
        0.3 * data["Rutting"] +
        0.2 * data["Cracks"] +
        0.1 * data["Potholes"]
    )

    def classify(pci):
        if pci >= 85:
            return "Good"
        elif pci >= 70:
            return "Fair"
        elif pci >= 50:
            return "Poor"
        else:
            return "Very Poor"

    data["Condition"] = data["PCI"].apply(classify)

    # ---------- SUMMARY ----------
    summary = data["Condition"].value_counts().to_dict()

    # ---------- MAP DATA ----------
    map_data = []
    if {"Lat", "Lon"}.issubset(data.columns):
        map_data = data[
            ["Lat", "Lon", "Lane", "Chainage", "PCI", "Condition"]
        ].to_dict(orient="records")

    # ---------- VIDEO ----------
    video_path = None
    video_ai_result = None

    if video_file and video_file.filename:
        video_path = video_file.filename
        full_video_path = os.path.join(app.config["UPLOAD_FOLDER"], video_path)
        video_file.save(full_video_path)

        # AI video analysis (safe)
        try:
            video_ai_result = analyze_video(full_video_path)
        except Exception as e:
            print("Video AI error:", e)
            video_ai_result = None

    return render_template(
        "dashboard.html",
        table=data.to_html(
            index=False,
            classes="table table-bordered table-striped table-hover text-center",
            border=0
        ),
        summary=summary,
        map_data=map_data,
        video_path=video_path,
        video_ai=video_ai_result,
        ai_result=latest_ai_result,
        filename=csv_file.filename
    )

# ------------------------
# DOWNLOAD CSV
# ------------------------
@app.route("/download/<filename>")
def download_file(filename):
    return send_file(
        os.path.join(app.config["UPLOAD_FOLDER"], filename),
        as_attachment=True
    )

# ------------------------
# AI IMAGE ANALYSIS
# ------------------------
@app.route("/ai-image", methods=["GET", "POST"])
def ai_image():
    global latest_ai_result

    if request.method == "POST":
        image = request.files.get("image")

        if not image or image.filename == "":
            return "No image uploaded"

        input_path = os.path.join(app.config["AI_UPLOAD_FOLDER"], image.filename)
        image.save(input_path)

        result = detect_road_damage(input_path)
        latest_ai_result = result

        return render_template(
            "ai_result.html",
            potholes=result["potholes"],
            cracks=result["cracks"],
            health_score=result["health_score"],
            condition=result["condition"],
            action=result["action"],
            image=result["output_image"]
        )

    return render_template("ai_upload.html")

# ------------------------
# TEST
# ------------------------
@app.route("/test")
def test():
    return "AI route working"

# ------------------------
# RUN
# ------------------------
if __name__ == "__main__":
    app.run(debug=True)
