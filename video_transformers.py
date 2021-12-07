""" 
Different transformers that can be applied to the video stream
"""

import cv2
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase


class EdgesVideoTransformer(VideoTransformerBase):
    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")

        # Apply canny algo for edge detection
        img = cv2.Canny(img, 100, 200)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        return img


