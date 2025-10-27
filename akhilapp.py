import streamlit as st
import os
import numpy as np
from PIL import Image
import time, base64, random
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

image_dir = "images"
audio_dir = "audio"
train_dir = "images"   # You must put sample training images in images/ANIMALNAME/

# Detect available animals
def detect_animals(image_dir):
    animals = []
    for folder in os.listdir(image_dir):
        folder_path = os.path.join(image_dir, folder)
        if os.path.isdir(folder_path) and len(os.listdir(folder_path)) > 0:
            animal = folder.rstrip("s") if folder.endswith("s") else folder
            animals.append(animal)
    return sorted(animals)

class_names = detect_animals(image_dir)

gallery_file_map = {}
audio_file_map = {}
for animal in class_names:
    img_folder = os.path.join(image_dir, animal + "s")
    gallery_image = None
    if os.path.exists(img_folder):
        images_found = [fname for fname in os.listdir(img_folder) if fname.lower().endswith(('.jpg', '.jpeg', '.png'))]
        if images_found:
            gallery_image = os.path.join(img_folder, images_found[0])
    gallery_file_map[animal] = gallery_image
    expected_audio = f"{animal}_roar.mp3"
    audio_path = os.path.join(audio_dir, expected_audio)
    if os.path.exists(audio_path):
        audio_file_map[animal] = audio_path

description = {
    "lion": "Lions are social big cats known as the 'King of the Jungle'.",
    "tiger": "Tigers are solitary big cats with striped fur and immense strength.",
    "cow": "Cows are domesticated mammals known for their calm nature and milk.",
    "elephant": "Elephants are large land mammals with trunks and tusks.",
    "rhino": "Rhinoceroses are large herbivores with thick skin and horns."
}
color_map = {
    "lion": "#FFD700",
    "tiger": "#FF8C00",
    "cow": "#8BC34A",
    "elephant": "#00BCD4",
    "rhino": "#9C27B0"
}

# Custom CSS
st.markdown("""
<style>
body { background: linear-gradient(to bottom right, #e3f2fd, #ffffff); }
h1, h2 { color: #1b3b6f; font-weight: bold; }
.stButton>button { background: linear-gradient(90deg, #f6d365, #fda085); color: white; font-size: 18px; border-radius: 15px; padding: 12px 25px; }
.stButton>button:hover { background: linear-gradient(90deg, #fda085, #f6d365); transform: scale(1.1); }
.card { background-color: white; padding: 25px; margin: 20px 0px; border-radius: 20px; box-shadow: 5px 5px 30px rgba(0,0,0,0.15); animation: fadeIn 1s ease-in-out; }
@keyframes fadeIn { 0% {opacity: 0;} 100% {opacity: 1;} }
</style>
""", unsafe_allow_html=True)

def autoplay_audio(file_path: str):
    with open(file_path, "rb") as f:
        data = f.read()
        b64 = base64.b64encode(data).decode()
        md = f"""<audio autoplay><source src="data:audio/mp3;base64,{b64}" type="audio/mp3"></audio>"""
        st.markdown(md, unsafe_allow_html=True)

st.sidebar.markdown("""
<div style='background-color:#fce4ec;padding:15px;border-radius:15px;text-align:center;'>
<h2 style='color:#d81b60;'>🐾 Animal Footprint Classifier (Train & Predict)</h2>
<p style='color:#880e4f;font-size:14px;'>Upload a footprint and predict the animal.</p>
</div>
""", unsafe_allow_html=True)
st.sidebar.markdown("---")
st.sidebar.markdown("""
<div style='background-color:#e0f7fa;padding:15px;border-radius:10px;'>
<h3 style='color:#00796b;'>How it works</h3>
<ol style='color:#004d40;font-size:14px;'>
<li>Train will use all images in each images/[animal]s folder!</li>
<li>Upload new footprint, select prediction algorithm.</li>
<li>See predicted animal, gallery, sound.</li>
</ol>
</div>
""", unsafe_allow_html=True)
st.sidebar.markdown("---")

st.title("🐾 Animal Footprint Classifier (Train & Predict)")
st.markdown("Upload a footprint image and select a prediction algorithm.")

uploaded_file = st.sidebar.file_uploader("Upload Footprint Image 🖼️", type=["png", "jpg", "jpeg"])
algo_choice = st.selectbox("Select Prediction Algorithm", ["Random Forest", "SVM","Random Choice"])

# Simple in-memory training
def load_images_and_labels(image_dir, animals, img_size=(64,64)):
    X, y = [], []
    for label, animal in enumerate(animals):
        folder_path = os.path.join(image_dir, animal + "s")
        for fname in os.listdir(folder_path):
            if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                img_path = os.path.join(folder_path, fname)
                img = Image.open(img_path).convert('RGB').resize(img_size)
                arr = np.array(img).flatten() / 255.0
                X.append(arr)
                y.append(label)
    return np.array(X), np.array(y)

if uploaded_file:
    img = Image.open(uploaded_file).convert('RGB')
    st.image(img.resize((250, 250)), caption='Uploaded Footprint', use_container_width=False)
    if st.button("Predict 🐾"):
        X, y = load_images_and_labels(train_dir, class_names)
        img_test = img.resize((64,64))
        x_in = np.array(img_test).flatten()[None,:] / 255.0
        # Train and predict
        if algo_choice == "Random Forest":
            clf = RandomForestClassifier(n_estimators=40, random_state=42)
            clf.fit(X, y)
            pred = clf.predict(x_in)[0]
            pred_class = class_names[pred]
            confidence = clf.predict_proba(x_in).max() * 100
        elif algo_choice == "SVM":
            clf = SVC(kernel='linear', probability=True)
            clf.fit(X, y)
            pred = clf.predict(x_in)[0]
            pred_class = class_names[pred]
            confidence = clf.predict_proba(x_in).max() * 100
        else:
            pred_class = random.choice(class_names)
            confidence = 100
        placeholder = st.empty()
        for i in range(5):
            placeholder.markdown(f"<h2 style='color:#1b3b6f;'>Predicting{'.'*i}</h2>", unsafe_allow_html=True)
            time.sleep(0.2)
        animal_info = description.get(pred_class, "No info available.")
        card_color = color_map.get(pred_class, "#222222")
        placeholder.markdown(f"""
        <div class='card' style='border-left:8px solid {card_color};'>
            <h2>Prediction: {pred_class.upper()} 🐾</h2>
            <p>Confidence: {confidence:.1f}%</p>
            <p>{animal_info}</p>
        </div>
        """, unsafe_allow_html=True)
        gallery_img = gallery_file_map.get(pred_class)
        audio_file = audio_file_map.get(pred_class)
        if gallery_img and os.path.exists(gallery_img):
            st.image(gallery_img, caption=pred_class.capitalize(), use_container_width=True)
        if audio_file and os.path.exists(audio_file):
            autoplay_audio(audio_file)

st.markdown("### 🖼️ Animal Gallery")
cols = st.columns(len(class_names))
for idx, name in enumerate(class_names):
    img_path = gallery_file_map.get(name)
    if img_path and os.path.exists(img_path):
        cols[idx].image(img_path, caption=name.title(), use_container_width=True)
