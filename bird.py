import sys
if __name__ == "__main__": 
    dataPath = sys.argv[1]
    trainStatus = sys.argv[2]
    modelPath = sys.argv[3] if len(sys.argv) > 3 else "default/model/path"

    print(f"Data path: {dataPath}")
    print(f"Train status: {trainStatus}")
    print(f"Model path: {modelPath}")

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torchvision import datasets, transforms
import torch.nn.functional as F
import torchvision
from torch.utils.data import random_split, DataLoader, Dataset, Subset

import os
import csv  # To handle CSV writing
from PIL import Image
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

# Define your transformations
train_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])


# Load the full dataset
full_dataset = datasets.ImageFolder(root=dataPath, transform=train_transform)

# Split dataset into training and validation sets
train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size
trainset, valset = torch.utils.data.random_split(full_dataset, [train_size, val_size])

# Apply validation transformation to the validation set
valset.dataset.transform = val_transform

# Create dataloaders
trainloader = DataLoader(trainset, batch_size=32, shuffle=True)
valloader = DataLoader(valset, batch_size=32, shuffle=False)

# Get the class labels
classes = full_dataset.classes

classes = ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9')

def imshow(img):
    img = img / 2 + 0.5 # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.axis("off")
    plt.show()

# get some random training images
n = 4
dataiter = np.random.choice(len(trainset), replace=False, size=n)
images, labels = [trainset[i][0] for i in dataiter], [trainset[i][1] for i in dataiter]

# show images
imshow(torchvision.utils.make_grid(images, padding=3))

print(labels)

# print labels
print(' '.join(f'{classes[labels[j]]:5s}' for j in range(n)))

class BirdDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert('RGB')
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
            
        return image, label

def conv_block(in_channels, out_channels, pool=False):
    layers = [nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1), 
              nn.BatchNorm2d(out_channels), 
              nn.ReLU(inplace=True)]
    if pool: layers.append(nn.MaxPool2d(2))
    return nn.Sequential(*layers)

class BirdClassifier(nn.Module):      
    def __init__(self, num_classes):
        super(BirdClassifier, self).__init__()

        self.conv1 = conv_block(3, 32)         # 64 x 32 x 32
        self.conv2 = conv_block(32, 64, pool=True)      # 128 x 16 x 16
        self.res1 = nn.Sequential(conv_block(64, 64), 
                                  conv_block(64, 64))  # 128 x 16 x 16

        self.conv3 = conv_block(64, 128, pool=True)    # 256 x 8 x 8
        self.conv4 = conv_block(128, 128, pool=True)    # 512 x 4 x 4
        self.res2 = nn.Sequential(conv_block(128, 128), 
                                  conv_block(128, 128)) 
        
        self.conv5 = conv_block(128, 256, pool=True)    # 256 x 8 x 8
        self.conv6 = conv_block(256, 256, pool=True)    # 512 x 4 x 4
        self.res3 = nn.Sequential(conv_block(256, 256), 
                                  conv_block(256, 256))  # 512 x 4 x 4
        
        self.classifier = nn.Sequential(nn.MaxPool2d(4), # 512 x 1 x 1
                                        nn.Flatten(),     # 512
                                        nn.Dropout(0.3),  
                                        nn.Linear(256, num_classes)) # 100


    def forward(self, xb):
        out1 = self.conv1(xb)
        out2 = self.conv2(out1)
        out3 = self.res1(out2) + out2
        out4 = self.conv3(out3)

        out5 = self.conv4(out4)

        out6 = self.res2(out5) + out5

        out7 = self.conv5(out6)

        out8 = self.conv6(out7)

        out9 = self.res3(out8) + out8

        out = self.classifier(out9)
        return out
    
    
import time
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class_counts = np.zeros(len(classes), dtype=int)
for _, label in trainset:
    class_counts[label] += 1

total_samples = sum(class_counts)
class_weights = [total_samples / count for count in class_counts]
print(class_weights)

class_weights_tensor = torch.tensor(class_weights, dtype=torch.float)

class_weights_tensor = class_weights_tensor.to(device)

# Instantiate the model and move it to the device
num_classes = len(classes)
model = BirdClassifier(num_classes=num_classes).to(device)
num_params = sum(p.numel() for p in model.parameters())
print(f"Total number of parameters: {num_params}")

# Define loss function and optimizer
# criterion = nn.CrossEntropyLoss()
criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
optimizer = optim.Adam(model.parameters(), lr=0.0005)


# Define a training function with a timeout
def train_with_timeout(model, trainloader, valloader, criterion, optimizer, num_epochs=10, max_runtime=6300):
    start_time = time.time()  # Start the timer
    best_val_loss = float("inf")  # Initialize best validation loss
    best_model_state = None  # Variable to store the best model

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        for images, labels in trainloader:
            images, labels = images.to(device), labels.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        # Print training loss and accuracy for each epoch
        print(f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {running_loss / len(trainloader):.4f}, Train Accuracy: {(correct * 100) / (32 * len(trainloader)):.4f}")

        # Validation step
        model.eval()
        val_loss = 0.0
        correct = 0

        with torch.no_grad():
            for images, labels in valloader:
                images, labels = images.to(device), labels.to(device)

                # Forward pass
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                _, predicted = torch.max(outputs, 1)
                correct += (predicted == labels).sum().item()

            avg_val_loss = val_loss / len(valloader)
            val_accuracy = (correct * 100) / (32 * len(valloader))
            print(f"Val Loss: {avg_val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")

        # Update the best model if current val loss is lower than best_val_loss
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_state = model.state_dict()
            print(f"New best model found at Epoch {epoch + 1} with Val Loss: {best_val_loss:.4f}")

        # Check if the training has exceeded the maximum runtime (2 hours)
        elapsed_time = time.time() - start_time
        if elapsed_time >= max_runtime:
            print("Time limit exceeded, stopping training.")
            break

    # Load the best model state
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"Training stopped. Best Val Loss: {best_val_loss:.4f}")
    else:
        print("Training completed within the time limit.")

    return best_val_loss

# Execute the training function with a 2-hour timeout
# train_with_timeout(model, trainloader, valloader, criterion, optimizer, num_epochs=25)

def validate(model, valloader, criterion):
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in valloader:
            images, labels = images.to(device), labels.to(device)
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    avg_loss = val_loss / len(valloader)
    accuracy = 100 * correct / total
    print(f"Validation Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")
    
    return avg_loss, accuracy

# Train or evaluate based on trainStatus
if trainStatus == "train":
    print("Training the model...")
    train_with_timeout(model, trainloader, valloader, criterion, optimizer, num_epochs=25)
    
    # Optionally, save the model
    torch.save(model.state_dict(), "bird.pth")
    print("Model saved as 'bird.pth'")

elif trainStatus == "test":
    print("Loading and inferring with the model...")
    
    # Load the pre-trained model from the specified path
    model.load_state_dict(torch.load(modelPath))
    model.eval()

    # Ensure the dataset for testing is set up
    test_dataset = datasets.ImageFolder(root=dataPath, transform=val_transform)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    def test_model(model, test_loader, device, output_csv='bird.csv'):
        model.eval()
        results = []
        with torch.no_grad():
            for images, _ in test_loader:
                images = images.to(device)
                
                # Forward pass
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                results.extend(predicted.cpu().numpy())
        
        # Write only predicted labels to a CSV file
        with open(output_csv, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Predicted_Label'])
            for label in results:
                writer.writerow([label])
        
        print(f"Predictions saved to {output_csv}")

    # Run inference and save the predictions
    test_model(model, test_loader, device)


