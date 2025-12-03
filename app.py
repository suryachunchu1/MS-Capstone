import os, json, io
import numpy as np
import streamlit as st
import tensorflow as tf
import cv2
from PIL import Image
from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten

# --------------------------- #
# Page + light theming        #
# --------------------------- #
st.set_page_config(page_title="Brain Tumor Classifier", page_icon="🧠", layout="wide")
st.markdown("""
<style>
:root { --card-bg:#ffffff; --soft:#f6f7fb; --border:#e9ecf3; }
.block-container { padding-top: 1rem; padding-bottom: 2rem; }
.card { background:var(--card-bg); border:1px solid var(--border); border-radius:16px; padding:18px; box-shadow:0 1px 2px rgba(0,0,0,.03); }
.kpi { display:flex; align-items:center; gap:10px; background:linear-gradient(180deg,#fafbff 0%,#f4f6ff 100%); border:1px solid var(--border); border-radius:14px; padding:14px 16px; }
.small { color:#6b7280; font-size:.9rem; }
.progress-wrap { background:#eef2ff; width:100%; height:10px; border-radius:999px; overflow:hidden; }
.progress-bar { height:100%; background:#3b82f6; border-radius:999px; }
.footer-note { color:#6b7280; font-size:.85rem; }
</style>
""", unsafe_allow_html=True)

# --------------------------- #
# Defaults / paths            #
# --------------------------- #
EF_MODEL_PATH = "models/efficientnet_b0_best.keras"
EF_LABELS_PATH = "models/labels.json"         # index->label mapping saved from training
CNN_MODEL_PATH = "models/custom_cnn_best.keras"

DEFAULT_LABELS = ['Glioma', 'Meningioma', 'No tumor', 'Pituitary']

# --------------------------- #
# Sidebar                     #
# --------------------------- #
with st.sidebar:
    st.header("⚙️ Settings")
    model_type = st.selectbox("Model", ["EfficientNetB0", "Custom CNN"])
    if model_type == "EfficientNetB0":
        ef_model_path = st.text_input("EfficientNet .keras", value=EF_MODEL_PATH)
        ef_labels_path = st.text_input("labels.json", value=EF_LABELS_PATH)
    else:
        cnn_model_path = st.text_input("CNN .keras / .h5", value=CNN_MODEL_PATH)
        st.caption("CNN uses the fixed label order: Glioma, Meningioma, No tumor, Pituitary.")
    st.divider()
    enable_cam = st.toggle("Enable Grad-CAM", value=True)
    cam_for_pred = st.toggle("Grad-CAM: use predicted class", value=True)
    manual_idx = st.text_input("Manual class index (optional)", value="")
    st.caption("If set, Grad-CAM targets this class instead of argmax.")
    st.divider()
    st.caption("Tip: keep model files in /models next to this app.")

# --------------------------- #
# Loaders                     #
# --------------------------- #
@st.cache_resource(show_spinner=True)
def load_efficientnet(model_path, labels_path):
    m = tf.keras.models.load_model(model_path)
    with open(labels_path, "r") as f:
        raw = json.load(f)                    # {"0":"glioma_tumor",...}
    labels = [raw[i] if isinstance(i, str) else raw[str(i)] for i in range(len(raw))]
    return m, labels

@st.cache_resource(show_spinner=True)
def load_cnn(model_path):
    try:
        m = load_model(model_path, compile=False)
    except Exception:
        # last-resort: basic head in case loading requires rebuild (weights may not load)
        m = Sequential([
            tf.keras.layers.Conv2D(32,(3,3),activation='relu',input_shape=(224,224,3)),
            tf.keras.layers.MaxPooling2D((2,2)),
            tf.keras.layers.Conv2D(64,(3,3),activation='relu'),
            tf.keras.layers.MaxPooling2D((2,2)),
            tf.keras.layers.Conv2D(128,(3,3),activation='relu'),
            tf.keras.layers.MaxPooling2D((2,2)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128,activation='relu'),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(4,activation='softmax'),
        ])
    return m

# --------------------------- #
# Preprocess per model type   #
# --------------------------- #
def preprocess_for(model_type, pil_img):
    if model_type == "EfficientNetB0":
        size = (224, 224)
        arr = np.array(pil_img.convert("RGB").resize(size), dtype=np.float32)
        arr = tf.keras.applications.efficientnet.preprocess_input(arr)
        return np.expand_dims(arr, 0), size
    else:
        size = (224, 224)
        arr = np.array(pil_img.convert("RGB").resize(size), dtype=np.float32) / 255.0
        return np.expand_dims(arr, 0), size

def predict(model, x):
    probs = model.predict(x, verbose=0)[0]
    idx = int(np.argmax(probs))
    return idx, float(probs[idx]), probs

# --------------------------- #
# Grad-CAM utilities          #
# --------------------------- #
def _detect_last_conv_layer(model):
    conv_like = (tf.keras.layers.Conv2D, tf.keras.layers.SeparableConv2D, tf.keras.layers.DepthwiseConv2D)
    try:
        return model.get_layer("top_conv")
    except Exception:
        pass
    for l in reversed(model.layers):
        if isinstance(l, conv_like):
            return l
    for l in reversed(model.layers):
        try:
            shape = l.output_shape
        except Exception:
            try:
                shape = tf.keras.backend.int_shape(l.output)
            except Exception:
                shape = None
        if shape is not None and isinstance(shape, tuple) and len(shape) == 4:
            return l
    return None

def gradcam(model, pil_img, size, target_index, preprocess_fn):
    layer = _detect_last_conv_layer(model)
    if layer is None:
        return None, None
    img = pil_img.convert("RGB").resize(size)
    arr = np.array(img, dtype=np.float32)
    arr = preprocess_fn(arr)
    x = np.expand_dims(arr, 0)

    grad_model = tf.keras.models.Model([model.inputs], [layer.output, model.output])
    with tf.GradientTape() as tape:
        conv_out, preds = grad_model(x)
        loss = preds[:, target_index]
    grads = tape.gradient(loss, conv_out)[0]
    weights = tf.reduce_mean(grads, axis=(0, 1, 2))
    cam = tf.reduce_sum(tf.multiply(weights, conv_out[0]), axis=-1).numpy()
    cam = np.maximum(cam, 0)
    cam = cam / (cam.max() + 1e-8)
    cam = cv2.resize(cam, size)
    heatmap = (255 * cam).astype(np.uint8)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    base = np.array(img)
    overlay = cv2.addWeighted(base, 0.60, heatmap, 0.40, 0)
    return Image.fromarray(overlay), layer.name

# --------------------------- #
# Saliency fallback           #
# --------------------------- #
def saliency_map(model, pil_img, size, class_index, preprocess_fn_for_sal):
    arr = np.array(pil_img.convert("RGB").resize(size), dtype=np.float32)
    arr = preprocess_fn_for_sal(arr)
    x = np.expand_dims(arr, 0)
    with tf.GradientTape() as tape:
        xt = tf.convert_to_tensor(x)
        tape.watch(xt)
        preds = model(xt)
        target = preds[:, class_index]
    grads = tape.gradient(target, xt)
    grads = tf.math.abs(grads)
    grads = tf.reduce_max(grads, axis=-1).numpy().squeeze()
    grads = cv2.resize(grads, size)
    grads = (grads - grads.min()) / (grads.max() - grads.min() + 1e-8)
    heat = (255 * grads).astype(np.uint8)
    heat = cv2.applyColorMap(heat, cv2.COLORMAP_JET)
    heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)
    base = np.array(pil_img.convert("RGB").resize(size))
    overlay = cv2.addWeighted(base, 0.60, heat, 0.40, 0)
    return Image.fromarray(overlay)

# --------------------------- #
# Load model + labels         #
# --------------------------- #
if model_type == "EfficientNetB0":
    if not (os.path.exists(ef_model_path) and os.path.exists(ef_labels_path)):
        st.error("Missing EfficientNet model or labels file.")
        st.stop()
    model, labels = load_efficientnet(ef_model_path, ef_labels_path)
    preprocess_model = lambda arr: tf.keras.applications.efficientnet.preprocess_input(arr)
else:
    if not os.path.exists(cnn_model_path):
        st.error("Missing CNN model file.")
        st.stop()
    model = load_cnn(cnn_model_path)
    labels = DEFAULT_LABELS
    preprocess_model = lambda arr: (arr / 255.0)

# --------------------------- #
# Header row                  #
# --------------------------- #
with st.container():
    cols = st.columns([1,1,1,2])
    with cols[0]:
        st.markdown(f"<div class='kpi'><div>📦</div><div><b>Model</b><br><span class='small'>{model_type}</span></div></div>", unsafe_allow_html=True)
    with cols[1]:
        st.markdown(f"<div class='kpi'><div>🏷️</div><div><b>Classes</b><br><span class='small'>{len(labels)}</span></div></div>", unsafe_allow_html=True)
    with cols[2]:
        last = _detect_last_conv_layer(model)
        lname = last.name if last is not None else "None"
        st.markdown(f"<div class='kpi'><div>🔥</div><div><b>Grad-CAM layer</b><br><span class='small'>{lname}</span></div></div>", unsafe_allow_html=True)

# --------------------------- #
# Upload + run                #
# --------------------------- #
st.subheader("Upload an MRI image")
uploaded = st.file_uploader("Drag & drop or browse", type=["png","jpg","jpeg"], label_visibility="collapsed")
if not uploaded:
    st.stop()

pil_img = Image.open(uploaded).convert("RGB")
x, size = preprocess_for(model_type, pil_img)
pred_idx, pred_prob, probs = predict(model, x)
pred_label = labels[pred_idx]

c1, c2 = st.columns([1,1], vertical_alignment="center")
with c1:
    st.markdown("<div class='card'><b>Original image</b>", unsafe_allow_html=True)
    st.image(pil_img, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

with c2:
    st.markdown("<div class='card'><b>Prediction</b>", unsafe_allow_html=True)
    st.markdown(f"<h3 style='margin:0'>{pred_label}</h3>", unsafe_allow_html=True)
    st.caption(f"Confidence: {pred_prob:.3f}")
    ordered = sorted([(labels[i], float(probs[i])) for i in range(len(probs))], key=lambda t: t[1], reverse=True)
    for name, p in ordered:
        st.markdown(f"<div style='display:flex;justify-content:space-between'><b>{name}</b><span>{p:.3f}</span></div>", unsafe_allow_html=True)
        st.markdown(f"<div class='progress-wrap'><div class='progress-bar' style='width:{max(0.0,min(1.0,p))*100:.2f}%'></div></div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

st.subheader("Explainability")

cam_img = None
layer_name = None
if enable_cam:
    target_idx = pred_idx
    if not cam_for_pred and manual_idx.strip().isdigit():
        target_idx = int(manual_idx.strip())
    cam_img, layer_name = gradcam(model, pil_img, size, target_idx, preprocess_model)

if cam_img is None:
    # Fallback saliency
    cam_img = saliency_map(
        model,
        pil_img,
        size,
        pred_idx,
        preprocess_model if model_type == "EfficientNetB0" else (lambda a: a/255.0)
    )

colx, coly = st.columns([1,1], vertical_alignment="center")
with colx:
    st.markdown("<div class='card'><b>Heatmap overlay</b>", unsafe_allow_html=True)
    st.image(cam_img, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)
with coly:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    if layer_name:
        st.caption(f"Grad-CAM layer: {layer_name}")
    else:
        st.caption("Grad-CAM unavailable → showing input-gradient saliency.")
    st.markdown("Red-yellow ≈ regions most influential for this class.")
    st.markdown("</div>", unsafe_allow_html=True)

st.write("")
st.markdown("<div class='footer-note'>For research/education only. Not a medical device.</div>", unsafe_allow_html=True)
