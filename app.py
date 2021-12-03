''' Simple streamlit webrtc page accessing the webcam
and looping back the vid stream
'''

import streamlit as st
#import cv2
from streamlit_webrtc import webrtc_streamer
from video_transformers import EdgesVideoTransformer

webrtc_streamer(
    key="edge_detection_vid_streamer",
    video_processor_factory=EdgesVideoTransformer
    )