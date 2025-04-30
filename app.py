import streamlit as st
from utils import download_video, extract_frames, load_model_from_pickle
import tempfile
import os
import numpy as np
import cv2

# Load pre-trained model
model = load_model_from_pickle('violence_model.pkl')  # Update path as needed

def predict_violence(video_url):
    """
    Predict violence in the video using pre-trained model
    """
    # Download the video from URL
    with tempfile.NamedTemporaryFile(delete=False) as temp_video:
        temp_video.close()
        download_video(video_url, temp_video.name)
        
        # Extract frames from the video
        frames = extract_frames(temp_video.name)
        
        # Process frames to get predictions
        results = []
        for frame in frames:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert to RGB for model
            frame = cv2.resize(frame, (224, 224))  # Resize to match model input size
            frame = np.expand_dims(frame, axis=0)  # Add batch dimension
            prediction = model.predict(frame)  # Get model prediction
            results.append(prediction)

        return results

def display_results(results):
    """
    Display the final verdict based on predictions
    """
    if results:
        # Flatten results and find the most frequent prediction (violent or non-violent)
        flattened_results = [item for sublist in results for item in sublist]
        final_verdict = max(set(flattened_results), key=flattened_results.count)
        st.write("✅ Final Verdict:", final_verdict)
    else:
        st.write("❌ No results to display.")

def main():
    """
    Main function to run the Streamlit app
    """
    st.title("Violence Detection in Videos")
    st.write("Enter the video URL to check for violence content")

    # Input field for video URL
    video_url = st.text_input("Video URL", "")

    if video_url:
        # Process video and predict
        st.write("Processing the video... This may take a while.")
        results = predict_violence(video_url)

        # Display results
        display_results(results)

if __name__ == "__main__":
    main()
