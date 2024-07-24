import torch
import torch.nn as nn
import torchvision.models as models
from PIL import Image
import torchvision.transforms as transforms

#Define densenet
class DenseNet(nn.Module):
    def __init__(self):
        super(DenseNet, self).__init__()
        self.densenet = models.densenet161(weights=models.DenseNet161_Weights.DEFAULT)
        self.densenet.classifier = nn.Identity()
    def forward(self,img):
        return self.densenet(img)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Function to load and preprocess images
def load_image(image_path, transform=transform):
    image = Image.open(image_path).convert('RGB')
    if transform:
        image = transform(image)
    image = image.unsqueeze(0)  # Add batch dimension
    return image

# Function to calculate Euclidean distance
def euclidean_distance(a, b):
    return torch.norm(a - b, dim=1)

def extract_vector(img_tensor):
    # feature extraction
    features = DenseNet()
    with torch.no_grad():
        vector = features(img_tensor)
    # normalization
    vector = vector / torch.norm(vector)
    return vector