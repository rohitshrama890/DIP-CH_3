import streamlit as st
import cv2
import numpy as np
from skimage.filters import sobel, prewitt, laplace
from skimage.segmentation import watershed
from skimage.color import rgb2gray
from PIL import Image

# Convert image to grayscale safely
def to_grayscale(image):
    """Ensure image is grayscale-ready."""
    if image.shape[-1] == 4:  # Convert RGBA to RGB
        image = image[..., :3]
    return rgb2gray(image) if image.shape[-1] == 3 else image

# Apply edge detection with scale factor
def apply_edge_detection(image, method, scale=1.0):
    gray = rgb2gray(image) if len(image.shape) == 3 else image
    gray = (gray * 255).astype(np.uint8)  # Convert to 8-bit grayscale

    if method == "Sobel":
        edges = sobel(gray)
    elif method == "Prewitt":
        edges = prewitt(gray)
    elif method == "Laplace (2nd Derivative)":
        edges = laplace(gray)
    elif method == "Canny":
        edges = cv2.Canny(gray, 50, 150)
    else:
        edges = gray  # Default to original image if no valid method

    # Normalize and convert to 8-bit for proper display
    if method in ["Sobel", "Prewitt", "Laplace (2nd Derivative)"]:
        edges = (edges - edges.min()) / (edges.max() - edges.min())  # Normalize to 0-1
        edges = (edges * 255).astype(np.uint8)  # Convert to 0-255

    # Apply scale factor
    edges = cv2.convertScaleAbs(edges, alpha=scale)  

    return edges

# Feature detection
def detect_features(image, feature_type, line_type=None, scale=1.0):
    gray = to_grayscale(image)
    gray = (gray * 255).astype(np.uint8)

    if feature_type == "Point Detection":
        kernel = np.array([[1, 1, 1], [1, -8, 1], [1, 1, 1]])
        result = cv2.filter2D(gray, -1, kernel)

    elif feature_type == "Line Detection":
        if line_type == "Vertical":
            kernel = np.array([[-1, 2, -1], [-1, 2, -1], [-1, 2, -1]])
        elif line_type == "Horizontal":
            kernel = np.array([[-1, -1, -1], [2, 2, 2], [-1, -1, -1]])
        elif line_type == "+45 Degree":
            kernel = np.array([[-1, -1, 2], [-1, 2, -1], [2, -1, -1]])
        elif line_type == "-45 Degree":
            kernel = np.array([[2, -1, -1], [-1, 2, -1], [-1, -1, 2]])
        else:
            return gray  # Default to original image if no valid selection

        result = cv2.filter2D(gray, -1, kernel)
    else:
        result = gray

    result = cv2.convertScaleAbs(result, alpha=scale)  # Apply scale factor
    return result

# Segmentation function
def segment_image(image, scale=1.0):
    gray = to_grayscale(image)
    edges = sobel(gray)
    markers = np.zeros_like(gray, dtype=np.int32)
    markers[gray < 0.3] = 1
    markers[gray > 0.7] = 2
    segmented = watershed(edges, markers)
    
    # Normalize and convert to displayable format
    segmented = (segmented * 255 / segmented.max()).astype(np.uint8)
    segmented = cv2.convertScaleAbs(segmented, alpha=scale)
    return segmented

# Streamlit App
st.title("🔍 Interactive Detection of Discontinuities")
st.write("Explore edge detection, feature detection, and segmentation interactively!")

# File Upload
uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = np.array(Image.open(uploaded_file).convert("RGB"))  # Convert RGBA to RGB
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    # Feature Detection UI
    st.subheader("🔎 Detect Points, Lines, and Edges")
    feature_type = st.selectbox("Choose Feature Detection Method", ["Point Detection", "Line Detection"])

    scale = st.slider("Adjust Scale", min_value=0.5, max_value=3.0, value=1.0, step=0.1)
    
    if feature_type == "Line Detection":
      line_type = st.radio("Select Line Discontinuity Type", ["Vertical", "Horizontal", "+45 Degree", "-45 Degree"])
      feature_image = detect_features(image, feature_type, line_type, scale)
    else:
      feature_image = detect_features(image, feature_type, scale=scale)

    st.image(feature_image, caption=f"{feature_type} Applied", use_column_width=True, clamp=True)

    # Edge Detection
    st.subheader("🖼️ Apply Edge Detection")
    method = st.selectbox("Choose Edge Detection Method", ["Sobel", "Prewitt", "Canny", "Laplace (2nd Derivative)"])
    processed_image = apply_edge_detection(image, method, scale)
    st.image(processed_image, caption=f"{method} Edge Detection", use_column_width=True, clamp=True)
    
    # Segmentation
    st.subheader("🔍 Apply Segmentation")
    segmented_img = segment_image(image, scale)
    st.image(segmented_img, caption="Segmented Image", use_column_width=True, clamp=True)
    
    # Quiz Section
    st.subheader("🎯 Quick Quiz: What did you learn?")
    quiz_q1 = st.radio("Which method detects edges by finding intensity gradients?", ("Watershed", "Sobel", "K-Means"))
    if quiz_q1 == "Sobel":
        st.success("✅ Correct!")
    else:
        st.error("❌ Try again!")
    
    quiz_q2 = st.radio("What is the purpose of segmentation?", ("Blurring the image", "Dividing an image into regions", "Changing colors"))
    if quiz_q2 == "Dividing an image into regions":
        st.success("✅ Correct!")
    else:
        st.error("❌ Try again!")

st.write("📌 This interactive app helps you explore image discontinuities!")
