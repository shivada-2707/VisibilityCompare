# app.py
import os
from flask import Flask, render_template, send_from_directory, abort
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

app = Flask(__name__, static_folder="static")

# --- CONFIGURE THESE PATHS TO MATCH YOUR PROJECT ---
BASE_RUNS = "/home/user/VisibilityProject/new_runs"   # root containing *_images and *_detect dirs
RAW_IMAGES = os.path.join(BASE_RUNS, "raw_images")
CLAHE_IMAGES = os.path.join(BASE_RUNS, "clahe_images")
AOD_IMAGES = os.path.join(BASE_RUNS, "aodnet_images")

RAW_LABELS = os.path.join(BASE_RUNS, "raw_detect", "run", "labels")
CLAHE_LABELS = os.path.join(BASE_RUNS, "clahe_detect", "run", "labels")
AOD_LABELS = os.path.join(BASE_RUNS, "aodnet_detect", "run", "labels")
# ---------------------------------------------------

# ensure static/plots exists
PLOTS_DIR = os.path.join(app.static_folder, "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

def read_labels_stats(labels_dir, stem):
    """Read YOLO style label file and return (count, conf_sum).
       If not found returns (0,0.0).
       We assume each line ends in a confidence/score (float)."""
    fname = os.path.join(labels_dir, f"{stem}.txt")
    if not os.path.isfile(fname):
        return 0, 0.0
    count = 0
    conf_sum = 0.0
    with open(fname, "r") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            # try parse last token as float (score)
            try:
                score = float(parts[-1])
            except Exception:
                score = 0.0
            count += 1
            conf_sum += score
    return count, conf_sum

def make_chart_counts(stem, values_dict, outpath):
    names = list(values_dict.keys())
    values = [values_dict[n] for n in names]
    colors = ["#1f77b4", "#ff9900", "#2ca02c"]  # blue, orange, green
    plt.figure(figsize=(6,4))
    bars = plt.bar(names, values, color=colors[:len(names)])
    plt.title(f"Detections - {stem}")
    plt.ylabel("Number of Detections")
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()

def make_chart_conf(stem, values_dict, outpath):
    names = list(values_dict.keys())
    values = [values_dict[n] for n in names]
    colors = ["#1f77b4", "#ff9900", "#2ca02c"]
    plt.figure(figsize=(6,4))
    plt.bar(names, values, color=colors[:len(names)])
    plt.title(f"Confidence Sum - {stem}")
    plt.ylabel("Confidence sum")
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()

def find_image_url(folder, stem):
    """Return a URL path to serve image if exists (tries png,jpg,jpeg), else None."""
    for ext in (".png", ".jpg", ".jpeg"):
        p = os.path.join(folder, stem + ext)
        if os.path.exists(p):
            # return the Flask route that serves this folder
            if folder == RAW_IMAGES:
                return f"/raw_images/{stem}{ext}"
            elif folder == CLAHE_IMAGES:
                return f"/clahe_images/{stem}{ext}"
            elif folder == AOD_IMAGES:
                return f"/aodnet_images/{stem}{ext}"
    return None

@app.route("/")
def index():
    # gather stems from RAW_IMAGES folder (png/jpg)
    stems = []
    if os.path.isdir(RAW_IMAGES):
        for fname in sorted(os.listdir(RAW_IMAGES)):
            base, ext = os.path.splitext(fname)
            if ext.lower() in (".png", ".jpg", ".jpeg"):
                stems.append(base)
    # unique and sorted
    stems = sorted(list(dict.fromkeys(stems)))
    return render_template("index.html", stems=stems)

@app.route("/compare/<stem>")
def compare(stem):
    stem = os.path.basename(stem)  # sanitize
    # read stats from label folders
    raw_count, raw_conf = read_labels_stats(RAW_LABELS, stem)
    clahe_count, clahe_conf = read_labels_stats(CLAHE_LABELS, stem)
    aod_count, aod_conf = read_labels_stats(AOD_LABELS, stem)

    # build charts in static/plots
    counts_fname = f"counts_{stem}.png"
    conf_fname = f"conf_{stem}.png"
    counts_path = os.path.join(PLOTS_DIR, counts_fname)
    conf_path = os.path.join(PLOTS_DIR, conf_fname)

    counts_dict = {"RAW": raw_count, "CLAHE": clahe_count, "AOD-Net": aod_count}
    conf_dict = {"RAW": raw_conf, "CLAHE": clahe_conf, "AOD-Net": aod_conf}

    # create charts (overwrite each time)
    make_chart_counts(stem, counts_dict, counts_path)
    make_chart_conf(stem, conf_dict, conf_path)

    # image URLs (served by below routes)
    raw_url = find_image_url(RAW_IMAGES, stem)
    clahe_url = find_image_url(CLAHE_IMAGES, stem)
    aod_url = find_image_url(AOD_IMAGES, stem)

    return render_template(
        "compare.html",
        stem=stem,
        raw_url=raw_url,
        clahe_url=clahe_url,
        aod_url=aod_url,
        raw_count=raw_count,
        raw_conf=raw_conf,
        clahe_count=clahe_count,
        clahe_conf=clahe_conf,
        aod_count=aod_count,
        aod_conf=aod_conf,
        chart_counts_url=f"/static/plots/{counts_fname}",
        chart_conf_url=f"/static/plots/{conf_fname}"
    )

# --- routes to serve images that are outside static/ ---
@app.route("/raw_images/<path:filename>")
def raw_images(filename):
    if not os.path.exists(os.path.join(RAW_IMAGES, filename)):
        abort(404)
    return send_from_directory(RAW_IMAGES, filename)

@app.route("/clahe_images/<path:filename>")
def clahe_images(filename):
    if not os.path.exists(os.path.join(CLAHE_IMAGES, filename)):
        abort(404)
    return send_from_directory(CLAHE_IMAGES, filename)

@app.route("/aodnet_images/<path:filename>")
def aodnet_images(filename):
    if not os.path.exists(os.path.join(AOD_IMAGES, filename)):
        abort(404)
    return send_from_directory(AOD_IMAGES, filename)

if __name__ == "__main__":
    # run server on all interfaces so you can access from other machines in LAN (your current setup)
    app.run(host="0.0.0.0", port=5000, debug=True)
