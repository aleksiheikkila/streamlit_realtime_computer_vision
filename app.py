''' Simple streamlit webrtc page accessing the webcam
and looping back the vid stream, possibly with some processing applied.
'''

import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode
from age_gender_estimator import AgeGenderPredVideoProcessor
from config import Config
#from video_transformers import EdgesVideoTransformer

conf = Config()

#webrtc_streamer(
#    key="edge_detection_vid_streamer",
#    video_processor_factory=EdgesVideoTransformer
#    )

st.write("""
    ## Real-time Age and Gender prediction

    Applies (not too great) CV models on the stream of incoming frames from client's camera,
    obtained thru webRTC. There are models for face detection, age and gender prediction.

    #### Open your camera
    """)

ctx = webrtc_streamer(
    key="age-gender-prediction", 
    video_processor_factory=AgeGenderPredVideoProcessor,
    mode=WebRtcMode.SENDRECV,
    media_stream_constraints={"video": True, "audio": False}
    )

if ctx.video_processor:
    ctx.video_processor.conf_threshold_face = st.slider(
        "Face detection confidence threshold", 0., 1.0, conf.conf_threshold_face
        ) 
    ctx.video_processor.conf_threshold_age = st.slider(
        "Age prediction confidence threshold", 0., 1.0, conf.conf_threshold_age
        )
    ctx.video_processor.conf_threshold_gender = st.slider(
        "Gender prediction confidence threshold", 0., 1.0, conf.conf_threshold_gender
        )