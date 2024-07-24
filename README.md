# CBIR - Find similar images

Here i used CBIR dataset includes nearly 4800 images for inference

## How it works

I used VGG16 to extract features from images and save them to a list of vectors. New image will go through this network and being compared to the existing vectors used Euclidean distance (L2 distance) and return n nearest images. 

## Examples
I use streamlit for displaying

![image](https://github.com/user-attachments/assets/5a37aa0e-3bf4-47d6-867f-2c22f6111099)

![image](https://github.com/user-attachments/assets/af6f39c0-fbb9-4752-a10e-be09c686f021)



 
