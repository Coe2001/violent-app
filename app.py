import streamlit as st
from utils import download_video, extract_frames, load_model_from_pickle, preprocess_frame
import tempfile
import os
import numpy as np

# Load model only once during the app initialization
@st.cache_resource
def load_model():
    return load_model_from_pickle("violence_model.pkl")

model = load_model()

def predict_violence(video_path):
    """
    Predict violence in a video using the pre-trained model.
    Args:
    - video_path: Path to the video file.
    
    Returns:
    - Prediction result (e.g., violence detected or not).
    """
    frames = extract_frames(video_path)
    predictions = []
    
    # Preprocess each frame and make predictions
    for frame in frames:
        processed_frame = preprocess_frame(frame)
        prediction = model.predict(processed_frame)
        predictions.append(prediction)

    # Return the most frequent prediction (violence or no violence)
    result = max(set(predictions), key=predictions.count)
    return "Violence Detected" if result == 1 else "No Violence Detected"

# Streamlit UI
st.title("Violence Detection in Video")
st.write("Upload a video to analyze and predict if there is any violence detected in it.")

# File upload widget
video_file = st.file_uploader("Upload Video", type=["mp4", "avi", "mov"])

if video_file:
    # Save the uploaded video to a temporary file
    temp_video_file = tempfile.NamedTemporaryFile(delete=False)
    temp_video_file.write(video_file.read())
    video_path = temp_video_file.name
    st.video(video_path)  # Display the video

    # Button to trigger the prediction
    if st.button("Predict Violence"):
        with st.spinner("Processing video..."):
            result = predict_violence(video_path)
            st.write(f"✅ Final Verdict: {result}")
            
    # Clean up the temporary file after prediction
    os.remove(video_path)
