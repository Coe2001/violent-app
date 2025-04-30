import streamlit as st
from utils import download_video, extract_frames, load_model_from_pickle, predict_frame
import tempfile
import os

st.title("🔍 Violence Detection from Video URL")

video_url = st.text_input("Paste a video URL (MP4 format recommended):")

if st.button("Analyze"):
    if video_url:
        with st.spinner("Downloading video..."):
            video_path = download_video(video_url)

        if video_path:
            with st.spinner("Extracting frames..."):
                frames = extract_frames(video_path)

            with st.spinner("Loading model..."):
                model = load_model_from_pickle("violence_model.pkl")

            st.success(f"Extracted {len(frames)} frames")

            results = []
            for i, frame in enumerate(frames):
                pred = predict_frame(model, frame)
                label = "Violence" if pred > 0.5 else "Non-Violence"
                results.append(label)
                st.image(frame, caption=f"Frame {i+1}: {label}", width=200)

            st.write("✅ Final Verdict:", max(set(results), key=results.count))
        else:
            st.error("Failed to download video.")
