import os
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.utils import image_dataset_from_directory
from tensorflow.keras import layers, models
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import joblib

img_size = (64,64)
data_dir = "dataset"
batch_size = 32

def load_images_for_ml(data_dir, img_size):
    X, y = [], []
    classes = sorted(os.listdir(data_dir))
    for label, class_folder in enumerate(classes):
        class_path = os.path.join(data_dir, class_folder)
        for fname in os.listdir(class_path):
            if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                img_path = os.path.join(class_path, fname)
                img = image.load_img(img_path, target_size=img_size)
                arr = image.img_to_array(img).flatten() / 255.0
                X.append(arr)
                y.append(label)
    return np.array(X), np.array(y), classes

X, y, class_names = load_images_for_ml(data_dir, img_size)

# Random Forest
rf = RandomForestClassifier(n_estimators=50, random_state=42)
rf.fit(X, y)
joblib.dump(rf, "animal_rf_model.pkl")

# SVM
svm = SVC(kernel='linear', probability=True)
svm.fit(X, y)
joblib.dump(svm, "animal_svm_model.pkl")

# CNN
train_ds = image_dataset_from_directory(data_dir, image_size=img_size, batch_size=batch_size, seed=42, validation_split=0.2, subset="training")
val_ds = image_dataset_from_directory(data_dir, image_size=img_size, batch_size=batch_size, seed=42, validation_split=0.2, subset="validation")

cnn = models.Sequential([
    layers.Rescaling(1./255, input_shape=img_size + (3,)),
    layers.Conv2D(32, (3,3), activation='relu'),
    layers.MaxPooling2D(2,2),
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D(2,2),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(len(class_names), activation='softmax')
])
cnn.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
cnn.fit(train_ds, validation_data=val_ds, epochs=10)
cnn.save("animal_classifier_model.h5")

with open("class_names.txt", "w") as f:
    f.write("\n".join(class_names))

print("All models trained and saved! CNN, Random Forest, SVM. Class names:", class_names)
