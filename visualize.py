from matplotlib import pyplot as plt
from model import load_image,extract_vector,euclidean_distance
import streamlit as st
import torch
import pickle
import math
from PIL import Image

vectors = pickle.load(open('vectors.pkl','rb'))
img_paths = pickle.load(open('img_paths.pkl','rb'))
num = 5
img_input = 'template.jpg'
img = load_image(img_input)
vector = extract_vector(img)
distance = euclidean_distance(vectors,vector)

ids = torch.argsort(distance)[:num]
nearest_image = [(img_paths[id], distance[id]) for id in ids]

axes = []
grid_size = int(math.sqrt(num))
fig = plt.figure(figsize=(10,5))

for id in range(num):
    draw_image = nearest_image[id]
    axes.append(fig.add_subplot(grid_size, grid_size, id+1))

    axes[-1].set_title(draw_image[1])
    plt.imshow(Image.open(draw_image[0]))

fig.tight_layout()
plt.show()

