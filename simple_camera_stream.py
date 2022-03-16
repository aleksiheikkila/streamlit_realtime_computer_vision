''' Simple streamlit webrtc page accessing the webcam
and looping back the vid stream
'''

import streamlit as st
from streamlit_webrtc import webrtc_streamer
#import cv2

webrtc_streamer(key="example_simple_vid_streamer")