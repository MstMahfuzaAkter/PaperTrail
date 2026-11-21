
import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io

st.set_page_config(
    page_title="PaperTrail",
    page_icon="📄",
    layout="wide"
)

st.title("📄 PaperTrail")
st.header("Document Boundary Detection & Restoration")

st.write(
    "An Image Processing Based Document Restoration System"
)


# --------------------------------
# FUNCTION 1: ORDER DOCUMENT POINTS
# --------------------------------

def order_points(points):

    rect = np.zeros((4, 2), dtype="float32")

    s = points.sum(axis=1)
    diff = np.diff(points, axis=1)

    rect[0] = points[np.argmin(s)]       # Top-left
    rect[2] = points[np.argmax(s)]       # Bottom-right
    rect[1] = points[np.argmin(diff)]    # Top-right
    rect[3] = points[np.argmax(diff)]    # Bottom-left

    return rect


# --------------------------------
# FUNCTION 2: DOCUMENT DETECTION
# --------------------------------

def detect_document(image):

    gray = cv2.cvtColor(
        image, cv2.COLOR_RGB2GRAY
    )

    blurred = cv2.GaussianBlur(
        gray, (5, 5), 0
    )

    edges = cv2.Canny(
        blurred, 50, 150
    )

    kernel = np.ones((5, 5), np.uint8)

    edges = cv2.dilate(
        edges, kernel, iterations=2
    )

    contours, _ = cv2.findContours(
        edges,
        cv2.RETR_LIST,
        cv2.CHAIN_APPROX_SIMPLE
    )

    contours = sorted(
        contours,
        key=cv2.contourArea,
        reverse=True
    )

    image_area = image.shape[0] * image.shape[1]

    for contour in contours:

        area = cv2.contourArea(contour)

        if area < image_area * 0.10:
            continue

        perimeter = cv2.arcLength(
            contour, True
        )

        approx = cv2.approxPolyDP(
            contour,
            0.02 * perimeter,
            True
        )

        if len(approx) == 4:

            return order_points(
                approx.reshape(4, 2).astype("float32")
            )

    return None


# --------------------------------
# FUNCTION 3: PERSPECTIVE TRANSFORM
# --------------------------------

def perspective_transform(image, points):

    tl, tr, br, bl = points

    width_a = np.linalg.norm(br - bl)
    width_b = np.linalg.norm(tr - tl)

    height_a = np.linalg.norm(tr - br)
    height_b = np.linalg.norm(tl - bl)

    max_width = max(
        1, int(max(width_a, width_b))
    )

    max_height = max(
        1, int(max(height_a, height_b))
    )

    destination = np.array([
        [0, 0],
        [max_width - 1, 0],
        [max_width - 1, max_height - 1],
        [0, max_height - 1]
    ], dtype="float32")

    matrix = cv2.getPerspectiveTransform(
        points, destination
    )

    warped = cv2.warpPerspective(
        image,
        matrix,
        (max_width, max_height),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255)
    )

    return warped


# --------------------------------
# FUNCTION 4: DOCUMENT RESTORATION
# --------------------------------

def restore_document(image):

    gray = cv2.cvtColor(
        image, cv2.COLOR_RGB2GRAY
    )

    denoised = cv2.fastNlMeansDenoising(
        gray,
        None,
        10,
        7,
        21
    )

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    enhanced = clahe.apply(denoised)

    binary = cv2.adaptiveThreshold(
        enhanced,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        21,
        10
    )

    return gray, enhanced, binary


# --------------------------------
# MAIN APPLICATION
# --------------------------------

uploaded_file = st.file_uploader(
    "Upload your document",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(
        uploaded_file
    ).convert("RGB")

    original = np.array(image)

    st.subheader("1. Original Document")

    st.image(
        original,
        caption="Original Uploaded Image",
        use_container_width=True
    )

    # Detect document corners
    points = detect_document(original)

    if points is not None:

        # Draw detected boundary
        boundary_image = original.copy()

        cv2.polylines(
            boundary_image,
            [points.astype(np.int32)],
            True,
            (0, 255, 0),
            5
        )

        st.subheader("2. Document Boundary Detection")

        st.image(
            boundary_image,
            caption="Detected Document Boundary",
            use_container_width=True
        )

        # Perspective correction
        corrected = perspective_transform(
            original, points
        )

        st.subheader("3. Perspective Correction")

        st.image(
            corrected,
            caption="Corrected Document",
            use_container_width=True
        )

        st.success(
            "Document boundary detected successfully!"
        )

    else:

        corrected = original.copy()

        st.warning(
            "Four document corners were not detected. "
            "Using the original image instead."
        )

    # Image restoration
    gray, enhanced, binary = restore_document(
        corrected
    )

    st.subheader("4. Image Restoration Results")

    col1, col2 = st.columns(2)

    with col1:

        st.image(
            gray,
            caption="Grayscale",
            use_container_width=True
        )

        st.image(
            enhanced,
            caption="Contrast Enhanced",
            use_container_width=True
        )

    with col2:

        st.image(
            binary,
            caption="Adaptive Thresholding",
            use_container_width=True
        )

    # Download
    st.subheader("5. Download Restored Document")

    output = io.BytesIO()

    Image.fromarray(corrected).save(
        output, format="PNG"
    )

    st.download_button(
        "Download Perspective Corrected Image",
        data=output.getvalue(),
        file_name="papertrail_corrected.png",
        mime="image/png"
    )

    st.success(
        "PaperTrail processing completed!"
    )
