import streamlit as st
import streamlit_webrtc as webrtc
from keras.models import load_model
from PIL import Image, ImageOps
import numpy as np
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Function to classify based on an image
def classify_waste(img):
    np.set_printoptions(suppress=True)

    model = load_model("keras_model.h5", compile=False)
    class_names = open("labels.txt", "r").readlines()

    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

    image = img.convert("RGB")
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)

    image_array = np.asarray(image)
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
    data[0] = normalized_image_array

    prediction = model.predict(data)
    index = np.argmax(prediction)
    class_name = class_names[index]
    confidence_score = prediction[0][index]

    # If the confidence is below 0.9, or the class name is unknown, classify as weeds
    if confidence_score < 0.9 or class_name not in [
        "0 Black Soybean\n",
        "1 Brown Soybean\n",
        "2 Canada Soybean\n",
        "3 Clsoy 1 Soybean\n",
        "4 Clsoy 2 Soybean\n",
        "5 Collection 1 Soybean\n",
        "6 Collection 2 Soybean\n",
        "7 Tiwala 10 Soybean\n",
    ]:
        class_name = "8 Weeds\n"

    return class_name, confidence_score

# Display the classification result based on the label
def display_classification_result(label, confidence_score):
    classification_map = {
        "0 Black Soybean\n": "BLACK SOYBEAN",
        "1 Brown Soybean\n": "BROWN SOYBEAN",
        "2 Canada Soybean\n": "CANADA SOYBEAN",
        "3 Clsoy 1 Soybean\n": "CLSOY 1 SOYBEAN",
        "4 Clsoy 2 Soybean\n": "CLSOY 2 SOYBEAN",
        "5 Collection 1 Soybean\n": "COLLECTION 1 SOYBEAN",
        "6 Collection 2 Soybean\n": "COLLECTION 2 SOYBEAN",
        "7 Tiwala 10 Soybean\n": "TIWALA 10 SOYBEAN",
        "8 Weeds\n": "WEEDS",
    }

    classification = classification_map[label]
    if classification == "WEEDS":
        st.warning(f"Classified as WEEDS with a confidence of {confidence_score:.2f}.")
    else:
        st.success(f"Classified as {classification} with a confidence of {confidence_score:.2f}.")

# Set up Streamlit page
st.set_page_config(layout='wide')
st.title("Soybean and Weeds Classifier App")

# Define a custom VideoTransformer class to handle webcam frames
class VideoTransformer(webrtc.VideoTransformerBase):
    def __init__(self):
        self.last_frame = None  # Store the latest frame

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")  # OpenCV uses BGR
        self.last_frame = img  # Store the frame
        return frame  # Return the frame for display

# Create Streamlit tabs for file upload and webcam capture
tab1, tab2 = st.tabs(["Upload an Image", "Capture from Webcam"])

# File upload tab
with tab1:
    st.header("Upload an Image")
    input_img = st.file_uploader("Upload your image", type=['jpg', 'png', 'jpeg'])

    if input_img is not None:
        if st.button("Classify"):
            col1, col2 = st.columns([1, 1])

            with col1:
                st.info("Your uploaded Image")
                st.image(input_img, use_column_width=True)

            with col2:
                st.info("Classification Result")
                image_file = Image.open(input_img)
                label, confidence_score = classify_waste(image_file)
                display_classification_result(label, confidence_score)

# Webcam capture tab with three columns
with tab2:
    st.header("Capture from Webcam")

    # Initialize session state for captured image
    if "captured_image" not in st.session_state:
        st.session_state["captured_image"] = None

    # Create three columns for webcam stream, capture button, and classify
    col1, col2, col3 = st.columns([2, 1, 1])

    with col1:
        # Drop-down to select front or back camera
        camera_option = st.selectbox(
            "Choose Camera",
            ["Front", "Back"]
        )

        # Camera constraints
        video_constraints = {
            "video": {
                "facingMode": "user" if camera_option == "Front" else "environment"
            }
        }

        # Stream the webcam with a unique key and video constraints
        webrtc_ctx = webrtc.webrtc_streamer(
            key="unique_webcam_stream",
            video_transformer_factory=VideoTransformer,
            media_stream_constraints=video_constraints,
            rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
        )

    with col2:
        # Capture image button and display captured image
        if webrtc_ctx.video_transformer:
            if st.button("Capture Image"):
                img_array = webrtc_ctx.video_transformer.last_frame  # Get the captured frame
                img = Image.fromarray(img_array, 'RGB')  # Convert to PIL Image

                # Store the captured image in session state
                st.session_state["captured_image"] = img

                # Display the captured image
                st.image(img, caption="Captured Image", use_column_width=True)

    with col3:
        # Classify button for the captured image
        if st.session_state["captured_image"] is not None:
            if st.button("Classify Captured Image"):
                label, confidence_score = classify_waste(st.session_state["captured_image"])
                display_classification_result(label, confidence_score)