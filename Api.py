import nest_asyncio
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse
import shutil
import os
import uvicorn
import cv2
import numpy as np
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.models import Model
from sklearn.metrics.pairwise import cosine_similarity

# Allow asyncio to work in Jupyter
nest_asyncio.apply()

# Load VGG16 Model for feature extraction
base_model = VGG16(weights='imagenet')
model = Model(inputs=base_model.input, outputs=base_model.get_layer('fc1').output)

def extract_frames(video_path, frame_interval=1):
    cap = cv2.VideoCapture(video_path)
    frames = []
    count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        if count % frame_interval == 0:
            frame = cv2.resize(frame, (224, 224))  
            frames.append(frame)
        
        count += 1

    cap.release()
    return np.array(frames)

def extract_features(frames):
    features = []
    for frame in frames:
        frame = np.expand_dims(frame, axis=0) 
        frame = preprocess_input(frame)       
        feature = model.predict(frame)
        features.append(feature.flatten())    
    
    return np.array(features)

def compare_videos(video_path1, video_path2, frame_interval=1):
    frames1 = extract_frames(video_path1, frame_interval)
    frames2 = extract_frames(video_path2, frame_interval)
    
    if len(frames1) != len(frames2):
        min_frames = min(len(frames1), len(frames2))
        frames1 = frames1[:min_frames]
        frames2 = frames2[:min_frames]

    features1 = extract_features(frames1)
    features2 = extract_features(frames2)

    similarities = []
    for feat1, feat2 in zip(features1, features2):
        sim = cosine_similarity([feat1], [feat2])[0][0]
        similarities.append(sim)
    
    average_similarity = np.mean(similarities)
    return average_similarity, features1, features2, similarities

# FastAPI App
app = FastAPI()

@app.get("/", response_class=HTMLResponse)
async def main():
    return """
    <html>
        <head>
            <title>Video Similarity Checker</title>
        </head>
        <body>
            <h3>Upload Two Videos to Compare Similarity</h3>
            <form action="/upload_videos/" enctype="multipart/form-data" method="post">
                <label for="file1">Video 1:</label>
                <input name="file1" type="file" accept="video/*">
                <br><br>
                <label for="file2">Video 2:</label>
                <input name="file2" type="file" accept="video/*">
                <br><br>
                <input type="submit">
            </form>
        </body>
    </html>
    """

@app.post("/upload_videos/")
async def upload_videos(file1: UploadFile = File(...), file2: UploadFile = File(...)):
    video1_path = f"temp_{file1.filename}"
    video2_path = f"temp_{file2.filename}"
    
    with open(video1_path, "wb") as buffer:
        shutil.copyfileobj(file1.file, buffer)
        
    with open(video2_path, "wb") as buffer:
        shutil.copyfileobj(file2.file, buffer)

    try:
        # Compare the videos and get similarity score and feature vectors
        similarity_score, features1, features2, similarities = compare_videos(video1_path, video2_path, frame_interval=30)
        
        # Cleanup temporary files
        os.remove(video1_path)
        os.remove(video2_path)
        
        # Create a detailed output for the feature vectors and similarity score
        feature_vectors_str1 = f"<h4>Feature Vectors for Video 1:</h4><pre>{np.array_str(features1)}</pre>"
        feature_vectors_str2 = f"<h4>Feature Vectors for Video 2:</h4><pre>{np.array_str(features2)}</pre>"
        similarity_vectors_str = f"<h4>Frame-by-frame Similarities:</h4><pre>{np.array_str(similarities)}</pre>"

        return HTMLResponse(
            f"<h3>Similarity score between videos: {similarity_score}</h3>" +
            feature_vectors_str1 +
            feature_vectors_str2 +
            similarity_vectors_str
        )
    
    except Exception as e:
        # Cleanup temporary files in case of error
        os.remove(video1_path)
        os.remove(video2_path)
        return HTMLResponse(f"<h3>Error: {str(e)}</h3>")

# Run FastAPI app inside Jupyter
def run_app():
    config = uvicorn.Config(app, host="127.0.0.1", port=8000, log_level="info")
    server = uvicorn.Server(config)
    server.run()

# Start the FastAPI app
run_app()
