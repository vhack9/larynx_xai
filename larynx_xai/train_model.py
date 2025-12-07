import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import resnet18
from easyfsl.samplers import TaskSampler
from easyfsl.methods import PrototypicalNetworks
from PIL import Image
import os
import random

class LaryngealDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []
        for label, subdir in enumerate(['healthy', 'cancer']):
            for img_name in os.listdir(os.path.join(root_dir, subdir)):
                self.samples.append((os.path.join(root_dir, subdir, img_name), label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        img = Image.open(img_path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, label

transform = transforms.Compose([
    transforms.Resize((100, 100)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

dataset = LaryngealDataset('data/', transform=transform)
train_size = int(0.8 * len(dataset))
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, len(dataset) - train_size])

N_WAY = 2  # Classes: healthy, cancer
N_SHOT = 5  # Shots per class
N_QUERY = 15  # Queries per class
N_EVALUATION_TASKS = 100

train_sampler = TaskSampler(train_dataset, n_way=N_WAY, n_shot=N_SHOT, n_query=N_QUERY, n_tasks=1000)
val_sampler = TaskSampler(val_dataset, n_way=N_WAY, n_shot=N_SHOT, n_query=N_QUERY, n_tasks=N_EVALUATION_TASKS)

train_loader = DataLoader(train_dataset, batch_sampler=train_sampler, num_workers=2, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_sampler=val_sampler, num_workers=2, pin_memory=True)

backbone = resnet18(pretrained=True)
backbone.fc = nn.Flatten()  # For few-shot

model = PrototypicalNetworks(backbone).cuda() if torch.cuda.is_available() else PrototypicalNetworks(backbone)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# Training loop (simplified)
for epoch in range(10):  # Adjust epochs
    model.train()
    for support_images, support_labels, query_images, query_labels, _ in train_loader:
        optimizer.zero_grad()
        model_output = model(support_images, support_labels, query_images)
        loss = criterion(model_output, query_labels)
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch}: Loss {loss.item()}")

# Evaluation
model.eval()
correct, total = 0, 0
for support_images, support_labels, query_images, query_labels, _ in val_loader:
    with torch.no_grad():
        model_output = model(support_images, support_labels, query_images)
        predicted = torch.argmax(model_output, 1)
        correct += (predicted == query_labels).sum().item()
        total += query_labels.size(0)
print(f"Accuracy: {correct / total * 100:.2f}%")

torch.save(model.state_dict(), 'best_model.pth')