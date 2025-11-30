import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from flask import Flask, render_template, request, url_for
import rasterio
from PIL import Image

# -----------------------------
# BASIC SETTINGS
# -----------------------------
IMG_SIZE = 64

CLASS_NAMES = [
    "AnnualCrop",
    "Forest",
    "HerbaceousVegetation",
    "Highway",
    "Industrial",
    "Pasture",
    "PermanentCrop",
    "Residential",
    "River",
    "SeaLake",
]

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "uploads"
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

# Folder for PNG previews of uploaded TIFs
PREVIEW_FOLDER = os.path.join("static", "previews")
os.makedirs(PREVIEW_FOLDER, exist_ok=True)

# -----------------------------
# LOAD MODELS
# -----------------------------
print("Loading models... this may take a bit.")
CNN_MODEL = keras.models.load_model("MODEL_CNN_13BANDS_V2.keras")
VGG_MODEL = keras.models.load_model("MODEL_VGG16_13BANDS_V2.keras")
print("Models loaded.")

# -----------------------------
# SCENE GEO-TIFF (for map clicks)
# -----------------------------
SCENE_TIF_PATH = "SCENE_13BANDS.tif"
SCENE_META = None  # bounds, width, height, crs

if os.path.exists(SCENE_TIF_PATH):
    try:
        with rasterio.open(SCENE_TIF_PATH) as src:
            bounds = src.bounds  # left, bottom, right, top
            SCENE_META = {
                "minx": bounds.left,
                "miny": bounds.bottom,
                "maxx": bounds.right,
                "maxy": bounds.top,
                "width": src.width,
                "height": src.height,
                "crs": src.crs.to_string() if src.crs else None,
                "count": src.count,
            }
        print("Scene metadata loaded:", SCENE_META)
    except Exception as e:
        print("Warning: could not read scene metadata:", e)
        SCENE_META = None
else:
    print("Scene GeoTIFF not found at", SCENE_TIF_PATH)


# -----------------------------
# HELPERS
# -----------------------------
def load_tif_13band(path):
    """Loads a GeoTIFF and returns (H, W, bands) float32."""
    with rasterio.open(path) as src:
        img = src.read()  # (bands, H, W)
    img = np.transpose(img, (1, 2, 0)).astype(np.float32)
    return img


def align_to_13_bands(img):
    """
    Ensure image has 13 bands.
    - If 12 bands: insert dummy band.
    - If >13: slice to first 13.
    - If <13: pad with zeros.
    """
    h, w, c = img.shape
    if c == 13:
        return img
    if c == 12:
        dummy = np.zeros((h, w, 1), dtype=img.dtype)
        img = np.concatenate([img[:, :, :10], dummy, img[:, :, 10:]], axis=-1)
        return img
    if c > 13:
        return img[:, :, :13]
    # c < 13
    pad = np.zeros((h, w, 13 - c), dtype=img.dtype)
    return np.concatenate([img, pad], axis=-1)


def preprocess_patch(image):
    """
    Resize to IMG_SIZE x IMG_SIZE and apply per-image min-max normalization.
    """
    image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))
    image = tf.cast(image, tf.float32)

    min_v = tf.reduce_min(image, axis=[0, 1], keepdims=True)
    max_v = tf.reduce_max(image, axis=[0, 1], keepdims=True)
    image = (image - min_v) / (max_v - min_v + 1e-8)

    return image


def make_rgb_preview(img, preview_filepath):
    """
    Better-looking RGB preview:
    - uses bands 4-3-2 (R-G-B)
    - 2–98 percentile stretch
    - gamma correction
    """
    try:
        rgb = img[:, :, [3, 2, 1]]
    except Exception:
        rgb = img[:, :, :3]

    rgb_reshaped = rgb.reshape(-1, 3)
    p2 = np.percentile(rgb_reshaped, 2, axis=0)
    p98 = np.percentile(rgb_reshaped, 98, axis=0)

    rgb_stretched = (rgb - p2) / (p98 - p2 + 1e-6)
    rgb_stretched = np.clip(rgb_stretched, 0, 1)

    gamma = 1 / 2.2
    rgb_gamma = np.power(rgb_stretched, gamma)

    rgb_uint8 = (rgb_gamma * 255).clip(0, 255).astype(np.uint8)
    Image.fromarray(rgb_uint8).save(preview_filepath)


def extract_patch_from_scene(lon, lat, patch_size=64):
    """
    Given lon/lat in WGS84, extract a (patch_size x patch_size x bands)
    patch from SCENE_13BANDS.tif and align to 13 bands.
    """
    if not os.path.exists(SCENE_TIF_PATH):
        raise RuntimeError("SCENE_13BANDS.tif not found")

    with rasterio.open(SCENE_TIF_PATH) as src:
        row, col = src.index(lon, lat)  # (row, col) in pixel space

        half = patch_size // 2
        row_off = row - half
        col_off = col - half

        # clamp so window stays inside image
        row_off = max(0, min(row_off, src.height - patch_size))
        col_off = max(0, min(col_off, src.width - patch_size))

        window = rasterio.windows.Window(col_off, row_off, patch_size, patch_size)
        patch = src.read(window=window)  # (bands, H, W)

    img = np.transpose(patch, (1, 2, 0)).astype(np.float32)  # (H, W, bands)
    img = align_to_13_bands(img)
    return img


def run_models_on_patch(img_tf, model_choice):
    """
    Given a tensor img_tf (1, H, W, 13) and model_choice, return prediction dict.
    """
    out = {
        "class_cnn": None,
        "probs_cnn": None,
        "class_vgg": None,
        "probs_vgg": None,
    }

    # CNN
    if model_choice in ("both", "cnn"):
        preds_cnn = CNN_MODEL.predict(img_tf, verbose=0)[0]
        idx_cnn = int(np.argmax(preds_cnn))
        out["class_cnn"] = CLASS_NAMES[idx_cnn]
        out["probs_cnn"] = sorted(
            [(CLASS_NAMES[i], float(preds_cnn[i])) for i in range(len(CLASS_NAMES))],
            key=lambda x: x[1],
            reverse=True,
        )

    # VGG
    if model_choice in ("both", "vgg"):
        preds_vgg = VGG_MODEL.predict(img_tf, verbose=0)[0]
        idx_vgg = int(np.argmax(preds_vgg))
        out["class_vgg"] = CLASS_NAMES[idx_vgg]
        out["probs_vgg"] = sorted(
            [(CLASS_NAMES[i], float(preds_vgg[i])) for i in range(len(CLASS_NAMES))],
            key=lambda x: x[1],
            reverse=True,
        )

    return out


# -----------------------------
# ROUTES
# -----------------------------
@app.route("/", methods=["GET", "POST"])
def index():
    # shared scene metadata for Leaflet
    scene_meta = SCENE_META

    # context for template
    ctx = dict(
        error=None,
        scene_meta=scene_meta,

        # upload results
        upload_filename=None,
        upload_model_choice="both",
        upload_class_cnn=None,
        upload_probs_cnn=None,
        upload_class_vgg=None,
        upload_probs_vgg=None,
        upload_preview_url=None,  # URL of RGB preview for uploaded TIF

        # map results
        map_clicked_lat=None,
        map_clicked_lon=None,
        map_model_choice="both",
        map_class_cnn=None,
        map_probs_cnn=None,
        map_class_vgg=None,
        map_probs_vgg=None,
    )

    if request.method == "POST":
        source = request.form.get("source", "upload")
        model_choice = request.form.get("model_choice", "both")

        # ------------------ A) Normal upload flow ------------------
        if source == "upload":
            ctx["upload_model_choice"] = model_choice

            if "file" not in request.files:
                ctx["error"] = "No file part in the request."
                return render_template("index.html", **ctx)

            file = request.files["file"]
            if file.filename == "":
                ctx["error"] = "No file selected. Please choose a .tif file."
                return render_template("index.html", **ctx)

            filepath = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
            file.save(filepath)
            ctx["upload_filename"] = file.filename

            try:
                img = load_tif_13band(filepath)
            except Exception as e:
                ctx["error"] = f"Error reading TIF file: {e}"
                return render_template("index.html", **ctx)

            img = align_to_13_bands(img)
            if img.shape[-1] != 13:
                ctx["error"] = f"Expected 13 bands after alignment, got {img.shape[-1]}."
                return render_template("index.html", **ctx)

            # --- create RGB preview PNG for the uploaded TIF (for display only) ---
            base_name = os.path.splitext(file.filename)[0]
            safe_base = base_name.replace(" ", "_")
            preview_filename = f"{safe_base}_preview.png"
            preview_filepath = os.path.join(PREVIEW_FOLDER, preview_filename)

            try:
                make_rgb_preview(img, preview_filepath)
                ctx["upload_preview_url"] = url_for(
                    "static", filename=f"previews/{preview_filename}"
                )
            except Exception as e:
                print("Warning: could not create preview image:", e)

            # preprocess and run models
            img_tf = preprocess_patch(img)
            img_tf = tf.expand_dims(img_tf, 0)

            preds = run_models_on_patch(img_tf, model_choice)
            ctx["upload_class_cnn"] = preds["class_cnn"]
            ctx["upload_probs_cnn"] = preds["probs_cnn"]
            ctx["upload_class_vgg"] = preds["class_vgg"]
            ctx["upload_probs_vgg"] = preds["probs_vgg"]

            return render_template("index.html", **ctx)

        # ------------------ B) Map-click flow ------------------
        elif source == "click":
            if SCENE_META is None:
                ctx["error"] = "Scene metadata not available. Is SCENE_13BANDS.tif present?"
                return render_template("index.html", **ctx)

            lat_str = request.form.get("click_lat")
            lon_str = request.form.get("click_lon")
            if not lat_str or not lon_str:
                ctx["error"] = "No map click coordinates received."
                return render_template("index.html", **ctx)

            try:
                lat = float(lat_str)
                lon = float(lon_str)
            except ValueError:
                ctx["error"] = "Invalid latitude/longitude values."
                return render_template("index.html", **ctx)

            ctx["map_clicked_lat"] = lat
            ctx["map_clicked_lon"] = lon
            ctx["map_model_choice"] = model_choice

            try:
                img = extract_patch_from_scene(lon, lat, patch_size=IMG_SIZE)
            except Exception as e:
                ctx["error"] = f"Error extracting patch from scene: {e}"
                return render_template("index.html", **ctx)

            if img.shape[-1] != 13:
                img = align_to_13_bands(img)

            img_tf = preprocess_patch(img)
            img_tf = tf.expand_dims(img_tf, 0)

            preds = run_models_on_patch(img_tf, model_choice)
            ctx["map_class_cnn"] = preds["class_cnn"]
            ctx["map_probs_cnn"] = preds["probs_cnn"]
            ctx["map_class_vgg"] = preds["class_vgg"]
            ctx["map_probs_vgg"] = preds["probs_vgg"]

            return render_template("index.html", **ctx)

    # GET: empty state
    return render_template("index.html", **ctx)


if __name__ == "__main__":
    app.run(debug=True)