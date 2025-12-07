import torch
from torch import nn
from torchvision import transforms
from torchvision.models import resnet18
from easyfsl.methods import PrototypicalNetworks
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from PIL import Image
import numpy as np
import cv2

# Load model
backbone = resnet18()
backbone.fc = nn.Flatten()
model = PrototypicalNetworks(backbone)
model.load_state_dict(torch.load('best_model.pth', map_location='cpu'))
model.eval()

transform = transforms.Compose([
    transforms.Resize((100, 100)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def predict_and_explain(image_path):
    img = Image.open(image_path).convert('RGB')
    input_tensor = transform(img).unsqueeze(0)

    # For few-shot inference: Simulate support (in prod, use few examples from DB)
    # Here, dummy support: Assume we have a few pre-loaded support images
    support_images = []  # Load 5 healthy + 5 cancer from data/ (implement loading)
    support_labels = torch.tensor([0]*5 + [1]*5)  # 0: healthy, 1: cancer
    support_images = torch.stack([transform(Image.open(p)) for p in support_paths])  # Define support_paths

    with torch.no_grad():
        output = model(support_images, support_labels, input_tensor)
        prob = torch.softmax(output, dim=1)
        pred_class = 1 if prob[0][1] > 0.5 else 0
        confidence = prob[0][pred_class].item()

    # Grad-CAM
    target_layers = [model.backbone.layer4[-1]]  # Last conv layer
    cam = GradCAM(model=model.backbone, target_layers=target_layers)  # Backbone for CAM
    grayscale_cam = cam(input_tensor=input_tensor, targets=[ClassifierOutputTarget(pred_class)])[0]
    rgb_img = np.array(img.resize((100, 100))) / 255.0
    visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
    heatmap_path = image_path.replace('.png', '_heatmap.png')
    cv2.imwrite(heatmap_path, visualization * 255)

    return "Early-Stage Cancer" if pred_class == 1 else "Healthy", confidence, heatmap_path