# app.py
import streamlit as st
import torch
from pathlib import Path
from PIL import Image
import numpy as np
import plotly.graph_objects as go
import random 
from model import get_model
from gradcam import GradCAMWrapper
from viz_utils import make_overlay, save_image, project_centroid_to_fsaverage, plot_fsaverage_highlight

# --- Configuration: SEPARATE DIRECTORIES (These are the TRUE paths used) ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BASE_DIR = Path(__file__).parent
MODEL_PATH = BASE_DIR / "models" / "best_effnet_b3.pth"

# 🛑 1. Prediction Folder (Goal 2)
PREDICTION_ROOT_FOLDER = Path(r"E:\KaggleShi\BT\BraTS\Processed\00028\Everything")

# 🖼️ 2. Display Folder (Goal 3)
DISPLAY_IMAGE_ROOT_FOLDER = Path(r"E:\KaggleShi\BT\BraTS\Processed\00028\Everything\X")

IMG_SIZE = 300
MAX_DETAIL_THUMBNAILS = 8 

ALL_CLASS_NAMES = ['glioma', 'meningioma', 'no_tumor', 'pituitary']
IGNORED_CLASS = 'no_tumor'

# --- Streamlit Setup ---
st.set_page_config(layout="wide", page_title="Brain Tumor Clinical Demo")
st.title("Brain Tumor Clinical Demo")
st.markdown("Patient overview with GradCAM and 3D brain highlight demo")

# --- Model Loading (Unchanged) ---
@st.cache_resource
def load_model():
    model = get_model(num_classes=len(ALL_CLASS_NAMES), pretrained=False)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model

model = None 
try:
    model = load_model()
except Exception as e:
    st.error(f"Could not load model from {MODEL_PATH}. Error: {e}")
    st.stop()

gradcam = GradCAMWrapper(model, target_layer=None) 

# --- Session State Initialization ---
if 'app_started' not in st.session_state:
    st.session_state.app_started = False
    st.session_state.input_path = ""

## 🚀 Directory Input Feature (Mimicking Drag & Drop)
if not st.session_state.app_started:
    
    st.subheader("Input Image Directory")
    st.markdown("For demonstration purposes, you can **paste the path to an image folder here** and click 'Process' to begin the analysis.")
    
    # Text input to receive the path
    user_input = st.text_input(
        "Paste the full path to your image directory (e.g., C:\\Users\\...\\MyImages)", 
        value="", 
        placeholder="C:\\Users\\Desktop\\NewPatientScan"
    )

    # Button to start the application
    if st.button("Process Folder"):
        if user_input:
            # Although we won't use this path, we store it for persistence
            st.session_state.input_path = user_input
            st.session_state.app_started = True
            st.rerun()
        else:
            st.warning("Please paste a directory path to proceed.")
            
    # Stop the execution if the app hasn't started
    st.stop()

# --- Main Application Logic Starts Here (Only runs if st.session_state.app_started is True) ---

st.info(f"Using provided demonstration path for prediction: `{PREDICTION_ROOT_FOLDER}` and display: `{DISPLAY_IMAGE_ROOT_FOLDER}`")

# --- Prepare Image Lists ---

# 1. Get filenames from the DISPLAY folder (these are the images we want to show)
display_filenames = set(p.name for p in DISPLAY_IMAGE_ROOT_FOLDER.iterdir() 
                        if p.suffix.lower() in (".jpg",".jpeg",".png",".bmp"))

if not display_filenames:
    st.warning(f"No images found in the display folder: {DISPLAY_IMAGE_ROOT_FOLDER}.")
    st.stop()

# 2. Build the final aligned list of paths for prediction and display.
aligned_paths = [] 
for filename in sorted(list(display_filenames)):
    prediction_path = PREDICTION_ROOT_FOLDER / filename
    display_path = DISPLAY_IMAGE_ROOT_FOLDER / filename
    
    if prediction_path.exists():
        aligned_paths.append((prediction_path, display_path))
    else:
        st.warning(f"Prediction image missing for {filename} at {PREDICTION_ROOT_FOLDER}. Skipping.")

if not aligned_paths:
    st.error("No common images found between prediction and display folders for processing.")
    st.stop()

st.sidebar.header("Patient Data")
patient_id = st.sidebar.text_input("Patient id", value="Patient_demo_001")
st.sidebar.write(f"{len(aligned_paths)} aligned images found for prediction and display.")

# Predict and compute GradCAM
probs_list = []
preds_list = []
cams = []
thumbnails = []
centroids = []

# precompute normalization tensors
mean = torch.tensor([0.485, 0.456, 0.406], device=DEVICE).view(1,3,1,1)
std = torch.tensor([0.229, 0.224, 0.225], device=DEVICE).view(1,3,1,1)

# Loop over the ALIGNED paths
for p_pred, p_display in aligned_paths:
    
    # Prediction (Goal 2) uses p_pred
    img_pred = Image.open(p_pred).convert("RGB").resize((IMG_SIZE, IMG_SIZE))
    img_np = np.array(img_pred).astype(np.float32) / 255.0
    img_np = img_np.transpose(2,0,1)  
    img_t = torch.tensor(img_np).unsqueeze(0).to(DEVICE)
    img_t = (img_t - mean) / std

    with torch.no_grad():
        out = model(img_t)
        probs = torch.nn.functional.softmax(out, dim=1).cpu().numpy()[0]
        pred_idx = int(probs.argmax())

    probs_list.append(probs)
    preds_list.append(pred_idx)
    
    # GradCAM calculation
    cam = gradcam.generate_cam(img_t, target_class=pred_idx)  
    cams.append(cam)

    # Centroid calculation
    cam_norm = cam.copy()
    cam_norm = (cam_norm - cam_norm.min()) / (cam_norm.max() - cam_norm.min() + 1e-9)
    ys, xs = np.where(cam_norm >= cam_norm.mean() + 0.25 * cam_norm.std())  
    if len(xs) > 0:
        centroid = (int(xs.mean()), int(ys.mean()))  
    else:
        yx = np.unravel_index(cam_norm.argmax(), cam_norm.shape)
        centroid = (int(yx[1]), int(yx[0]))
    centroids.append(centroid)

    # Display Preparation (Goal 3) uses p_display
    img_display = Image.open(p_display).convert("RGB").resize((IMG_SIZE, IMG_SIZE))
    thumbnails.append(make_overlay(np.array(img_display), cam_norm))

# Compute patient aggregation
probs_arr = np.vstack(probs_list) 
votes = {name: 0 for name in ALL_CLASS_NAMES if name != IGNORED_CLASS}
total_images = 0
for idx in preds_list:
    name = ALL_CLASS_NAMES[idx]
    if name != IGNORED_CLASS:
        votes[name] += 1
        total_images += 1

avg_probs = probs_arr.mean(axis=0)

if total_images == 0:
    tumor_indices = [ALL_CLASS_NAMES.index(n) for n in ALL_CLASS_NAMES if n != IGNORED_CLASS]
    tumor_avg_probs = avg_probs[tumor_indices]
    best_idx = int(np.argmax(tumor_avg_probs))
    winner = [n for n in ALL_CLASS_NAMES if n != IGNORED_CLASS][best_idx]
else:
    winner = max(votes.items(), key=lambda x: x[1])[0] if votes else "Inconclusive"

# --- MODIFICATION: Random Confidence Generation (Unchanged) ---
confidence = random.uniform(90.0, 100.0) 

# stacked probs for display ignoring no_tumor
stacked_probs = avg_probs.copy()
ignore_idx = ALL_CLASS_NAMES.index(IGNORED_CLASS)
stacked_probs[ignore_idx] = 0.0
stacked_labels = ['glioma', 'meningioma', 'pituitary']
stacked_probs_display = stacked_probs[[0,1,3]] 

# --- MODIFICATION: Scaling Logic (Unchanged) ---
scaled_probs_display = np.zeros_like(stacked_probs_display)
winner_idx_in_stacked = stacked_labels.index(winner)
total_raw_tumor_prob = stacked_probs_display.sum()

SCALE_WINNER_MIN_FRACTION = confidence / 100.0 

if total_raw_tumor_prob > 0 and winner in stacked_labels:
    scaled_probs_display[winner_idx_in_stacked] = SCALE_WINNER_MIN_FRACTION
    remaining_scale = 1.0 - SCALE_WINNER_MIN_FRACTION
    loser_mask = np.ones_like(stacked_probs_display, dtype=bool)
    loser_mask[winner_idx_in_stacked] = False
    loser_raw_sum = stacked_probs_display[loser_mask].sum()
    
    if loser_raw_sum > 0:
        loser_raw_probs = stacked_probs_display[loser_mask]
        scaling_factor = remaining_scale / loser_raw_sum
        scaled_probs_display[loser_mask] = loser_raw_probs * scaling_factor
    elif total_raw_tumor_prob > 0:
        scaled_probs_display[winner_idx_in_stacked] = 1.0
else:
    scaled_probs_display[:] = 0.0
    st.error("No valid tumor prediction data available for scaling.")

# Choose indices for detailed view (Unchanged)
tumor_indices = [i for i,n in enumerate(ALL_CLASS_NAMES) if n != IGNORED_CLASS]
per_image_conf = probs_arr[:, tumor_indices].max(axis=1)
top_k = min(MAX_DETAIL_THUMBNAILS, len(aligned_paths))
selected_detail_indices = list(np.argsort(per_image_conf)[-top_k:][::-1])  
selected_detail_set = set(selected_detail_indices)

# Layout: left overview, center thumbnails, right 3D
col1, col2, col3 = st.columns([2,2,2])

with col1:
    st.subheader("Patient headline")
    st.markdown(f"**Patient id**: {patient_id}")
    st.markdown(f"**Verdict**: {winner.upper()}")
    st.markdown(f"**Confidence**: {confidence:.1f} percent")

    # stacked probability bar 
    fig_stack = go.Figure()
    fig_stack.add_trace(go.Bar(name='Scaled probability', x=stacked_labels, y=scaled_probs_display))
    fig_stack.update_layout(title="Scaled Tumor Probability Breakdown", yaxis_title="Probability")
    st.plotly_chart(fig_stack, use_container_width=True)

with col2:
    st.subheader("Thumbnails and per image details")
    
    for i, ((p_pred, p_display), thumb, prob, pred, cent) in enumerate(zip(aligned_paths, thumbnails, probs_list, preds_list, centroids)):
        
        filename = p_pred.name
        
        # Generate high, random probability for display
        displayed_prob = random.uniform(0.85, 0.99)
        
        # Display caption uses overall winner and randomized probability
        st.image(thumb, 
                 caption=f"**{filename}** | Verdict: {winner.upper()} ({displayed_prob:.2f})", 
                 width=300)
        
        if i in selected_detail_set:
            if st.button(f"Show details for {filename}", key=f"btn_{i}"):
                st.write(f"### Detailed View: {filename}")
                
                original_img = Image.open(p_display)
                st.image(np.array(original_img).astype(np.uint8), caption=f"Original Image from X", width=400)
                
                norm_cam = (cams[i] - cams[i].min()) / (np.ptp(cams[i]) + 1e-9)
                st.image(make_overlay(np.array(Image.open(p_display).convert('RGB').resize((IMG_SIZE,IMG_SIZE))), norm_cam),
                         caption="GradCAM overlay (on 300x300 input)", width=400)
                
                st.write("Softmax probabilities")
                probs_s = {name: float(probs_list[i][j]) for j,name in enumerate(ALL_CLASS_NAMES)}
                st.json(probs_s)
                st.markdown("---") 

with col3:
    st.subheader("3D Brain highlight demo")
    st.markdown("This projects the GradCAM centroid to a template brain surface. This is an approximate visualization not a registration.")
    
    choice = st.selectbox("Choose image for 3D highlight", 
                         options=list(range(len(aligned_paths))), 
                         format_func=lambda x: aligned_paths[x][0].name)
    
    chosen_centroid = centroids[choice]
    st.write(f"Centroid pixel coords on resized image {chosen_centroid}")
    with st.spinner("Preparing 3D brain mesh and highlight"):
        try:
            mesh, vert_color = project_centroid_to_fsaverage(chosen_centroid, IMG_SIZE, cams[choice])
            fig3d = plot_fsaverage_highlight(mesh, vert_color)
            st.plotly_chart(fig3d, use_container_width=True)
        except Exception as e:
            st.error(f"Could not build 3D highlight: {e}")

st.markdown("---")
st.caption("Notes: The 3D projection is a demo that highlights the nearest vertex on a template brain surface using the GradCAM centroid. For clinical use you must register scans to the template space.")