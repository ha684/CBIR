from model import load_image, extract_vector, euclidean_distance
import streamlit as st
import torch
import pickle
from PIL import Image

# Load data
vectors = pickle.load(open('vectors.pkl', 'rb'))
vectors = torch.cat(vectors, dim=0)
img_paths = pickle.load(open('img_paths.pkl', 'rb'))

# Streamlit app
st.title("Image Similarity Search")

# File uploader for input image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Display original image
    st.subheader("Original Image")
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Process the image
    img = load_image(uploaded_file)
    vector = extract_vector(img)
    distance = euclidean_distance(vectors, vector)
    
    # Number of similar images to display
    num = st.slider("Number of similar images to display", min_value=1, max_value=10, value=3)
    
    ids = torch.argsort(distance)[:num]
    nearest_images = [img_paths[id.item()] for id in ids]

    # Display similar images
    st.subheader(f"Top {num} Similar Images")
    cols = st.columns(num)
    for i, img_path in enumerate(nearest_images):
        with cols[i]:
            img = Image.open(img_path)
            st.image(img, caption=f"Similar Image {i+1}", use_column_width=True)

else:
    st.write("Please upload an image to start the similarity search.")