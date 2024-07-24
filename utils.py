import pickle
from model import load_image,extract_vector
import os

data_folder = '../../../data/dataset'

img_paths = [os.path.join(data_folder,img) for img in os.listdir(data_folder)]
vectors = [extract_vector(load_image(img_path)) for img_path in img_paths]

print('Loading...')
if not os.path.exists('vectors.pkl'):
    with open('vectors.pkl','wb') as f:
        pickle.dump(vectors,f)

if not os.path.exists('img_paths.pkl'):
    with open('img_paths.pkl','wb') as f:
        pickle.dump(img_paths,f)
        
print('Done')
