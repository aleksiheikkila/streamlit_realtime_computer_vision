### Streamlit web app accessing camera thru webrtc

Testing streamlit with WebRTC.

Receives incoming ~real-time video frames from client's camera (via WebRTC). Applies three CV models on the frame:
 - One to detect faces, getting the bounding boxes
 - One to predict age for each face (classify into a set of bins)
 - One to predict gender for each face

The approach is based on this [tutorial](https://learnopencv.com/age-gender-classification-using-opencv-deep-learning-c-python/). Also the used models are from that work and come with a bunch of known caveats.

In this case, a similar kind of approach is applied on the real-time stream of frames, instead of still images.

The models are not state of the art. The idea for this repo was just to monkey around toward an website that accesses local camera stream and applies computer vision stuff on that - so just to get the "framework" working, not to build something fancy.

Run locally with: 
> streamlit run app.py

The website/app is also accessible for the time being in [Heroku](https://age-and-gender.herokuapp.com/). It is just very slow, but gives the idea.
