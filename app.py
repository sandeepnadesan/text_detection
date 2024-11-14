import streamlit as st
import numpy as np
import cv2
from PIL import Image

def text_detect(img, ele_size=(8, 2)):
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_sobel = cv2.Sobel(img, cv2.CV_8U, 1, 0)
    _, img_threshold = cv2.threshold(img_sobel, 0, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY)
    element = cv2.getStructuringElement(cv2.MORPH_RECT, ele_size)
    img_threshold = cv2.morphologyEx(img_threshold, cv2.MORPH_CLOSE, element)
    res = cv2.findContours(img_threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    
    if cv2.__version__.startswith('3'):
        _, contours, hierarchy = res
    else:
        contours, hierarchy = res
    
    Rect = [cv2.boundingRect(i) for i in contours if i.shape[0] > 100]
    RectP = [
        (
            int(i[0] - i[2] * 0.08),
            int(i[1] - i[3] * 0.08),
            int(i[0] + i[2] * 1.1),
            int(i[1] + i[3] * 1.1)
        ) for i in Rect
    ]
    return RectP

def main():
    st.title("Text Detection in Image")
    
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # Read and process the image
        image = np.array(Image.open(uploaded_file))
        
        # Display the original uploaded image
        st.image(image, caption="Uploaded Image", use_column_width=True)
        
        # Detect text regions
        rects = text_detect(image)
        
        # Draw rectangles on the image
        for rect in rects:
            cv2.rectangle(image, rect[:2], rect[2:], (0, 0, 255), 2)
        
        # Display the processed image with detected text regions
        st.image(image, caption="Detected Text Regions", use_column_width=True)

if __name__ == "__main__":
    main()
