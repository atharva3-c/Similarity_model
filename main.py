import sqlite3
from fastapi import FastAPI, HTTPException, File, UploadFile, Form
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
import numpy as np
import cv2
from sklearn.metrics.pairwise import cosine_similarity
import tempfile
import os

# SQLite Database File Path
DB_PATH = "vectors.db"

app = FastAPI()

# Load the VGG16 model
base_model = VGG16(weights='imagenet')
model = Model(inputs=base_model.input, outputs=base_model.get_layer('fc1').output)

# Database Initialization Function
def initialize_database():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS video_features (
            video_id TEXT PRIMARY KEY,
            feature_vector TEXT
        );
    """)
    conn.commit()
    cursor.close()
    conn.close()

# SQLite Connection
def get_db_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

# Extract exactly 11 frames from video bytes
def extract_frames_from_bytes(video_bytes, target_frame_count=11):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_video:
        temp_video.write(video_bytes)
        temp_video_path = temp_video.name

    cap = cv2.VideoCapture(temp_video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_indices = np.linspace(0, total_frames - 1, target_frame_count, dtype=int)
    frames = []

    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (224, 224))
        frames.append(frame)

    cap.release()
    os.remove(temp_video_path)

    # If fewer than target frames were captured, duplicate the last frame
    while len(frames) < target_frame_count:
        frames.append(frames[-1])

    # Debug: Print actual number of frames extracted
    print(f"Extracted {len(frames)} frames.")
    
    return np.array(frames[:target_frame_count])

# Extract a consistent feature vector of shape (11, 4096)
def extract_feature_vector(video_bytes):
    frames = extract_frames_from_bytes(video_bytes, target_frame_count=11)
    
    if frames.shape[0] != 11:
        raise ValueError(f"Expected 11 frames but got {frames.shape[0]} frames.")
    
    features = []
    for frame in frames:
        frame = np.expand_dims(frame, axis=0)
        frame = preprocess_input(frame)
        feature = model.predict(frame)
        
        # Verify feature shape and append
        if feature.shape == (1, 4096):
            features.append(feature.flatten())
        else:
            raise ValueError(f"Unexpected feature shape: {feature.shape}, expected (1, 4096)")

    feature_matrix = np.array(features)
    
    # Debug: Confirm final feature matrix shape
    print(f"Feature matrix shape: {feature_matrix.shape}")
    
    if feature_matrix.shape != (11, 4096):
        raise ValueError(f"Feature matrix shape is {feature_matrix.shape}, expected (11, 4096)")

    return feature_matrix


# Fetch all feature vectors from the database
def get_all_feature_vectors():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT video_id, feature_vector FROM video_features")
    results = cursor.fetchall()
    cursor.close()
    conn.close()
    return [(video_id, np.array(eval(feature_vector)).reshape(11, 4096)) for video_id, feature_vector in results]

# Insert a new feature vector into the database
def insert_new_feature_vector(video_id, feature_matrix):
    flattened_vector = feature_matrix.flatten().tolist()
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        """
        INSERT INTO video_features (video_id, feature_vector)
        VALUES (?, ?)
        """,
        (video_id, str(flattened_vector))
    )
    conn.commit()
    cursor.close()
    conn.close()
    return video_id

# Compare the video feature vector with existing vectors in the database
def compare_with_fixed_vector(file_id, Fv_fixed, threshold=0.5):
    all_feature_vectors = get_all_feature_vectors()
    max_similarity = -1
    most_similar_video_id = None

    for video_id, feature_vector in all_feature_vectors:
        frame_similarities = [cosine_similarity([f_fixed], [f])[0][0] for f_fixed, f in zip(Fv_fixed, feature_vector)]
        max_frame_similarity = max(frame_similarities)

        if max_frame_similarity > max_similarity:
            max_similarity = max_frame_similarity
            most_similar_video_id = video_id

    if max_similarity < threshold:
        new_video_id = insert_new_feature_vector(file_id, Fv_fixed)

    return most_similar_video_id, max_similarity

@app.on_event("startup")
def startup_event():
    initialize_database()

@app.post("/compare-video-bytes/")
async def compare_video(file_id: str = Form(...), file: UploadFile = File(...)):
    try:
        video_bytes = await file.read()
        features_fixed = extract_feature_vector(video_bytes)
        video_id, max_similarity = compare_with_fixed_vector(file_id, features_fixed)

        return {"video_id": video_id, "max_similarity": max_similarity}
    except Exception as e:
        print(f"Error during video comparison: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error. Please check logs.")
