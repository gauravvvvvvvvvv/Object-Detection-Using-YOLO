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

@st.cache_resource
def load_model():
    return YOLO_Pred('./Model/weights/best.onnx', 'data.yaml')

class YOLOTransformer(VideoTransformerBase):
    def __init__(self, model):
        self.model = model

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        result_img, _ = self.model.predictions(img)  # Apply YOLO predictions
        return result_img  # Return the result without color conversion

def process_image(image, model):
    image_np = np.array(image)
    image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)  # Convert to BGR
    result_image, _ = model.predictions(image_np)
    return result_image

def process_camera(model):
    ctx = webrtc_streamer(
        key="camera",
        video_transformer_factory=lambda: YOLOTransformer(model),
        async_transform=True,
        mode=WebRtcMode.SENDRECV,
        client_settings=WEBRTC_CLIENT_SETTINGS
    )

    if ctx.state.playing:
        st.write("Camera is running!")
        while True:
            try:
                frame = ctx.video_transformer.transform(ctx.video_frame)
                st.image(frame, channels="BGR")
            except Exception as e:
                st.error("Error processing video frame: " + str(e))


def process_camera(model):
    ctx = webrtc_streamer(
        key="camera",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration={
            "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
        },
        video_transformer_factory=lambda: YOLOTransformer(model),
        async_transform=True,
        client_settings=WEBRTC_CLIENT_SETTINGS
    )

    if ctx.state.playing:
        st.write("Camera is running!")
    else:
        st.write("Camera is not running.")

def main():
    st.title("Object Detection with YOLO")

    # Load the YOLO model
    model = load_model()

    # Input type selection
    input_type = st.radio("Select input type:", ("Image", "Video", "Camera"))

    if input_type == "Image":
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            result = process_image(image, model)
            
            # Display the result
            st.image(result, caption='Processed Image', use_column_width=True, channels="BGR")
    
    elif input_type == "Video":
        uploaded_file = st.file_uploader("Choose a video...", type=["mp4", "avi", "mov"])
        if uploaded_file is not None:
            process_video(uploaded_file, model)
            st.text("Video processing complete!")
    
    elif input_type == "Camera":
        process_camera(model)

if __name__ == "__main__":
    main()
