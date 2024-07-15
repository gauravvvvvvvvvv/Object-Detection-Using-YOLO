import streamlit as st
import cv2
import numpy as np
from PIL import Image
import tempfile
from streamlit_webrtc import VideoTransformerBase, webrtc_streamer, WebRtcMode

# Import your YOLO_Pred class here
from yolo_predictions import YOLO_Pred

WEBRTC_CLIENT_SETTINGS = {
    "rtcConfiguration": {
        "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
    },
    "mediaStreamConstraints": {
        "video": True,
        "audio": False,
    },
}

@st.cache_resource()
def load_model():
    return YOLO_Pred('./Model/weights/best.onnx', 'data.yaml')

class YOLOTransformer(VideoTransformerBase):
    def __init__(self, model):
        self.model = model
        self.confidence_threshold = 0.4  # Initial confidence threshold

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        result_img, _ = self.model.predictions(img, self.confidence_threshold)  # Apply YOLO predictions
        return result_img  # Return the result without color conversion

def process_image(image, model, confidence_threshold):
    image_np = np.array(image)
    image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)  # Convert to BGR
    result_image, _ = model.predictions(image_np, confidence_threshold)
    return result_image

def process_video(video_file, model, confidence_threshold):
    tfile = tempfile.NamedTemporaryFile(delete=False) 
    tfile.write(video_file.read())

    vf = cv2.VideoCapture(tfile.name)
    stframe = st.empty()
    while vf.isOpened():
        ret, frame = vf.read()
        if not ret:
            break
        result = process_image(frame, model, confidence_threshold)
        stframe.image(result, channels="BGR")
        
    vf.release()
    
def main():
    st.title("Object Detection with YOLO")

    # Load the YOLO model
    model = load_model()

    # Show balloons only once per session
    if st.session_state.get('first_load', True):
        st.balloons()
        st.session_state['first_load'] = False

    # Confidence threshold slider
    confidence_threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.4, 0.05)
    st.text("Low for more detections. High for accuracy.")

    # Input options
    st.title("Input Options")
    input_type = st.selectbox("Select input type:", options=["Image", "Video"], index=0)

    if input_type == "Image":
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            result = process_image(image, model, confidence_threshold)

            # Display the result
            st.image(result, caption='Processed Image', use_column_width=True, channels="BGR")

    elif input_type == "Video":
        uploaded_file = st.file_uploader("Choose a video...", type=["mp4", "avi", "mov"])
        if uploaded_file is not None:
            process_video(uploaded_file, model, confidence_threshold)
            st.text("Video processing complete!")

if __name__ == "__main__":
    main()
