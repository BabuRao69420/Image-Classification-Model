import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models

# Load the CIFAR-10 dataset/apply transformations 
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False)

# Load the trained model
resnet = models.resnet18(pretrained=True)
resnet.fc = torch.nn.Linear(resnet.fc.in_features, 10)  # Modify the last fully connected layer for CIFAR-10

# Load weights
resnet.load_state_dict(torch.load('resnet_model.pth'))

# Evaluation
correct = 0
total = 0

with torch.no_grad():
    resnet.eval()  # Set the model to evaluation mode
    for images, labels in testloader:
        outputs = resnet(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = correct / total
print(f"Accuracy on the test set: {100 * accuracy:.2f}%")
