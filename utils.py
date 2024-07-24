import pickle
from model import load_image,extract_vector
import os

# Main processing
print('Starting...')

data_folder = '../../../data/dataset'
img_paths = [os.path.join(data_folder, img) for img in os.listdir(data_folder) if img.lower().endswith(('.png', '.jpg', '.jpeg'))]

vectors = []
for img_path in img_paths:
    print(f'Processing {img_path}')
    img = load_image(img_path)
    vector = extract_vector(img)
    vectors.append(vector.cpu())  # Move to CPU for saving

print('Extraction complete. Saving results...')

# Save vectors and image paths
with open('vectors.pkl', 'wb') as f:
    pickle.dump(vectors, f)

with open('img_paths.pkl', 'wb') as f:
    pickle.dump(img_paths, f)

print('Done')
