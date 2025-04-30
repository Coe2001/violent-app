import os
import cv2
import joblib
import requests
import numpy as np

def download_video(video_url, destination_path):
    """
    Download the video from the URL and save it to the given path.
    Args:
    - video_url: URL of the video to be downloaded.
    - destination_path: Path where the video will be saved.
    """
    response = requests.get(video_url, stream=True)
    if response.status_code == 200:
        with open(destination_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=1024):
                if chunk:
                    f.write(chunk)
        print(f"Video downloaded to {destination_path}")
    else:
        raise Exception(f"Failed to download video. Status code: {response.status_code}")

def extract_frames(video_path):
    """
    Extract frames from a video file.
    Args:
    - video_path: Path to the video file.
    
    Returns:
    - A list of frames extracted from the video.
    """
    # Open video file using OpenCV
    video = cv2.VideoCapture(video_path)
    frames = []
    
    # Loop through all frames in the video
    while True:
        ret, frame = video.read()
        if not ret:
            break
        frames.append(frame)  # Add frame to list

    video.release()  # Release the video object
    return frames

def load_model_from_pickle(model_path):
    """
    Load the pre-trained model from a pickle file.
    Args:
    - model_path: Path to the saved model (pickle file).
    
    Returns:
    - Loaded machine learning model.
    """
    try:
        # Load model using joblib
        model = joblib.load(model_path)
        print(f"Model loaded from {model_path}")
        return model
    except Exception as e:
        raise Exception(f"Error loading model: {e}")

def preprocess_frame(frame):
    """
    Preprocess a frame for model prediction.
    Args:
    - frame: The raw frame extracted from the video.
    
    Returns:
    - Preprocessed frame ready for model input.
    """
    # Resize the frame to the model's input size
    frame_resized = cv2.resize(frame, (224, 224))
    
    # Convert frame from BGR to RGB
    frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
    
    # Normalize the pixel values to [0, 1]
    frame_normalized = frame_rgb / 255.0
    
    # Expand dimensions to match the model input (batch_size, height, width, channels)
    frame_expanded = np.expand_dims(frame_normalized, axis=0)
    
    return frame_expanded
