import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models

# Define the data transformation
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomCrop((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Load the CIFAR-10 training dataset and apply transformations
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)

# Split into training and validation sets
train_size = int(0.8 * len(trainset))
val_size = len(trainset) - train_size
trainset, valset = torch.utils.data.random_split(trainset, [train_size, val_size])

# Load the CIFAR-10 testing dataset and apply transformations
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

# Create data loaders for training, validation, and testing
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
valloader = torch.utils.data.DataLoader(valset, batch_size=64, shuffle=False)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False)

# Define the pre-trained ResNet model
resnet = models.resnet18(pretrained=True)

# Modify the last fully connected layer for CIFAR-10 (10 output classes)
num_classes = 10
resnet.fc = torch.nn.Linear(resnet.fc.in_features, num_classes)

# Define the loss function and optimizer
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(resnet.parameters(), lr=0.001, momentum=0.9)

# Training loop
num_epochs = 1

for epoch in range(num_epochs):
    for images, labels in trainloader:
        # Forward pass
        outputs = resnet(images)
        loss = criterion(outputs, labels)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Print training progress
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

# Save the trained model weights
torch.save(resnet.state_dict(), 'resnet_model.pth')

# Evaluation
correct = 0
total = 0

with torch.no_grad():
    for images, labels in testloader:
        outputs = resnet(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = correct / total
print(f"Accuracy on the test set: {100 * accuracy:.2f}%")
