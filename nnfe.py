import torch
import torch.nn as nn
import torchvision.transforms as transforms

class FacialExpressionNet(nn.Module):
    def __init__(self):
        super(FacialExpressionNet, self).__init__()
        
        # Define the convolutional layers
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        
        # Define the max pooling layers
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Define the fully connected layers
        self.fc1 = nn.Linear(in_features=128 * 8 * 8, out_features=512)
        self.fc2 = nn.Linear(in_features=512, out_features=7)
        
        # Define the dropout layer
        self.dropout = nn.Dropout(p=0.2)
        
    def forward(self, x):
        # Pass the input through the convolutional layers
        x = self.pool(nn.functional.relu(self.conv1(x)))
        x = self.pool(nn.functional.relu(self.conv2(x)))
        x = self.pool(nn.functional.relu(self.conv3(x)))
        
        # Flatten the output of the convolutional layers
        x = x.view(-1, 128 * 8 * 8)
        
        # Pass the output through the fully connected layers
        x = self.dropout(nn.functional.relu(self.fc1(x)))
        x = self.fc2(x)
        
        return x

# Initialize the model and set it to the device
model = FacialExpressionNet()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Define the loss function and optimizer
criterion = nn.CrossEnt
