import os
import cv2
import numpy as np
from tqdm import tqdm
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.applications.xception import Xception, preprocess_input
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras.optimizers import Adam

# ===================== CONFIG ===================== #
DATA_PATH = r"C:\Users\azoos\OneDrive\Desktop\AI DEEP FORGE\FaceForensics-master"
FRAME_DIR = 'frames'  # Output directory for extracted frames
FRAME_SIZE = (299, 299)
MAX_FRAMES = 30
BATCH_SIZE = 16
EPOCHS = 5
DATASET_PATHS = {
    'original': 'original_sequences/youtube',
    'Deepfakes': 'manipulated_sequences/Deepfakes',
    'Face2Face': 'manipulated_sequences/Face2Face',
    'FaceSwap': 'manipulated_sequences/FaceSwap'
}

COMPRESSION = 'c0'  # Use 'c23' or 'c40' if needed
# ================================================== #

def extract_frames(video_path, output_folder, max_frames=MAX_FRAMES):
    os.makedirs(output_folder, exist_ok=True)
    reader = cv2.VideoCapture(video_path)
    frame_num = 0
    while reader.isOpened() and frame_num < max_frames:
        success, image = reader.read()
        if not success:
            break
        frame = cv2.resize(image, FRAME_SIZE)
        cv2.imwrite(os.path.join(output_folder, f"frame_{frame_num:03d}.jpg"), frame)
        frame_num += 1
    reader.release()

def extract_method_videos(data_path, dataset):
    print(f"[INFO] Extracting frames for: {dataset}")
    videos_path = os.path.join(data_path, DATASET_PATHS[dataset], COMPRESSION, 'videos')
    label = 'real' if dataset == 'original' else 'fake'
    for video in tqdm(os.listdir(videos_path)):
        name = os.path.splitext(video)[0]
        output_folder = os.path.join(FRAME_DIR, label, name)
        if not os.path.exists(output_folder):
            extract_frames(os.path.join(videos_path, video), output_folder)

def build_model():
    base = Xception(weights='imagenet', include_top=False, input_shape=(299, 299, 3))
    x = base.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation='relu')(x)
    out = Dense(1, activation='sigmoid')(x)
    model = Model(inputs=base.input, outputs=out)
    model.compile(optimizer=Adam(1e-4), loss='binary_crossentropy', metrics=['accuracy'])
    return model

def train_model():
    datagen = ImageDataGenerator(preprocessing_function=preprocess_input, validation_split=0.2)
    train_gen = datagen.flow_from_directory(
        FRAME_DIR,
        target_size=FRAME_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='binary',
        subset='training'
    )
    val_gen = datagen.flow_from_directory(
        FRAME_DIR,
        target_size=FRAME_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='binary',
        subset='validation'
    )
    model = build_model()
    model.fit(train_gen, validation_data=val_gen, epochs=EPOCHS)
    model.save("deepfake_model.h5")
    print("[INFO] Model saved as deepfake_model.h5")
    return model

def predict_image(img_path, model_path='deepfake_model.h5'):
    model = load_model(model_path, compile=False)
    img = load_img(img_path, target_size=FRAME_SIZE)
    img = img_to_array(img)
    img = preprocess_input(img)
    img = np.expand_dims(img, axis=0)
    prediction = model.predict(img)[0][0]
    return "Fake" if prediction > 0.5 else "Real", float(prediction)

if __name__ == '__main__':
    # Step 1: Extract frames
    extract_method_videos(DATA_PATH, 'original')     # Real videos
    extract_method_videos(DATA_PATH, 'Deepfakes')    # Fake videos

    # Step 2: Train model with real and fake frames
    model = train_model()

    # Step 3: Predict a test image (modify the path to any real or fake frame)
    test_image = 'frames/fake/baby.png'  # Update with a valid frame path
    if os.path.exists(test_image):
        result, score = predict_image(test_image)
        print(f"[INFO] Prediction: {result} (Confidence: {score:.2f})")
    else:
        print(f"[WARNING] Test image not found: {test_image}")

