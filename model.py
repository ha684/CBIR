import torch
import torch.nn as nn
import torchvision.models as models
from PIL import Image
import torchvision.transforms as transforms

# Define DenseNet
class VGG16(nn.Module):
    def __init__(self):
        super(VGG16, self).__init__()
        vgg16 = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
        self.features = vgg16.features
        self.avgpool = vgg16.avgpool
        self.classifier = nn.Sequential(*list(vgg16.classifier.children())[:-1])  # Remove last fully connected layer

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

# Initialize the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = VGG16().to(device)
model.eval()  # Set the model to evaluation mode

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
    return image.to(device)

# Function to calculate Euclidean distance
def euclidean_distance(a, b):
    return torch.norm(a.to(device) - b.to(device), dim=1)

# Function to extract feature vector
def extract_vector(img_tensor):
    img_tensor = img_tensor.to(device)
    with torch.no_grad():
        vector = model(img_tensor)
    # Normalization (for each vector if it's a batch)
    vector = vector / torch.norm(vector, dim=1, keepdim=True)
    return vector