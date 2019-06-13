#---------------------------------------------
# Import necessary packages
#---------------------------------------------

import pandas as pd
import numpy as np
import time
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler 
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from scipy.io import loadmat

#---------------------------------------------
# Spambase Data
#--------------------------------------------- 

# Hyper-parameters 
input_size = 57
hidden_size = 50
num_classes = 2
num_epochs = 5
batch_size = 10
learning_rate = 0.01

# Define a customized class for spambase data
class SpamDataset(Dataset):
    """Customized class that pre-processes spam dataset"""
    def __init__(self, csv_file):
        """
        Args:
            csv_file (string): Name of dataset file
        """
        # Read in data
        self.landmarks_frame = pd.read_csv(csv_file, header = None)
        
        # Standardize data
        m = np.mean(self.landmarks_frame.iloc[:,0:57].values)
        st = np.std(self.landmarks_frame.iloc[:,0:57].values)
        self.landmarks_frame.iloc[:,0:57] = (self.landmarks_frame.iloc[:,0:57]-m)/st

    def __len__(self):
        return len(self.landmarks_frame)

    def __getitem__(self, idx):
        # Get the features and make it into tensor
        features = self.landmarks_frame.iloc[idx, 0:57].values
        features = np.array([[features]])
        features = torch.from_numpy(features)
        # Get the label
        label = self.landmarks_frame.iloc[idx, -1]
        # Combine features and label into a tuple
        sample = (features,label)
        return sample

# Get dataset
dataset = SpamDataset("spambase.data")

# Specify portion for splitting, shuffle data, set seed
test_split = .2
shuffle_dataset = True
random_seed= 5

# Creating data indices for training and testing splits
dataset_size = len(dataset)
indices = list(range(dataset_size))
split = int(np.floor(test_split * dataset_size))
if shuffle_dataset :
    np.random.seed(random_seed)
    np.random.shuffle(indices)
train_indices, val_indices = indices[split:], indices[:split]

# Creating data samplers and loaders

train_sampler = SubsetRandomSampler(train_indices)
test_sampler = SubsetRandomSampler(val_indices)

train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, 
                                           sampler=train_sampler)
test_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                                sampler=test_sampler)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') #CUDA is a GPU

# Convolutional neural network (two convolutional layers)
class ConvNet(nn.Module):
    def __init__(self, num_classes=10):
        super(ConvNet, self).__init__()
        torch.manual_seed(10) # For reproducibility
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.fc = nn.Linear(14*64, num_classes)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        return out

# Set model
model = ConvNet(num_classes).double().to(device)

# Set loss function and optimize algorithm
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)  

# Set start time
start = time.time()


for epoch in range(num_epochs):
    train_loss, test_loss = [], []
    correct = 0
    total = 0
    # Train the model
    for i, (x, y) in enumerate(train_loader):  
        # Move tensors to the configured device
        x = x.to(device)
        y = y.to(device)
        
        # Forward pass
        outputs = model(x)
        loss = criterion(outputs, y)
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss.append(loss.item())
    
    with torch.no_grad():
        
        # Predict the model
        for x, y in test_loader:
            x = x.to(device)
            y = y.to(device)
            outputs = model(x)
            lose = criterion(outputs, y)
            _, predicted = torch.max(outputs.data, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()
            test_loss.append(lose.item())

        print ("Epoch:", epoch + 1, ", Training Loss: ", np.mean(train_loss), ", Test loss: ", np.mean(test_loss))

# Set end time        
end = time.time()  
print('Accuracy is: {} %'.format(100 * correct / total))
print('It takes ' +  str(end - start)  + ' seconds to run Feedforward Neural Network algorithm on spambase dataset.')

#---------------------------------------------
# Breast Cancer Data
#---------------------------------------------

# Hyper-parameters 
input_size = 30
hidden_size = 20
num_classes = 2
num_epochs = 5
batch_size = 10
learning_rate = 0.01

# Read in data
df = pd.read_csv("wdbc.data",header = None)

# Drop patients' ID column
df.drop(df.columns[[0]], axis=1, inplace=True)

# Change categorical labels to binary labels
categorical_features = [1]
df[1] = LabelEncoder().fit_transform(df[1])

# Make a class for breast cancer dataset
class BCData(Dataset):
    """Customized class that pre-processes breast cancer dataset"""
    def __init__(self, data):
        """
        Args:
            data: The breast cancer pandas dataframe 
        """
        self.main = data
        self.n = len(data)

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        features = self.main.iloc[idx,1:31].values
        features = features.astype('double')
        features = np.array([[features]])
        features = torch.from_numpy(features)
        
            
        label = self.main.iloc[idx,0]

        sample = (features,label)


        return sample

# Get the dataset through the class
dataset = BCData(df)

# Specify portion for splitting, shuffle data, set seed
percent_split = .2
shuffle_dataset = True
random_seed= 5

# Creating data indices for training and testing splits
dataset_size = len(dataset)
indices = list(range(dataset_size))
split = int(np.floor(percent_split * dataset_size))
if shuffle_dataset :
    np.random.seed(random_seed)
    np.random.shuffle(indices)
train_indices, val_indices = indices[split:], indices[:split]

# Creating PT data samplers and loaders
train_sampler = SubsetRandomSampler(train_indices)
test_sampler = SubsetRandomSampler(val_indices)

train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, 
                                           sampler=train_sampler)
test_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                                sampler=test_sampler)

# Convolutional neural network (two convolutional layers)
class ConvNet(nn.Module):
    def __init__(self, num_classes=10):
        super(ConvNet, self).__init__()
        torch.manual_seed(7) # For reproducibility
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.fc = nn.Linear(7*64, num_classes)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        return out

# Set model
model = ConvNet(num_classes).double().to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)  

# Set start time
start = time.time()


for epoch in range(num_epochs):
    train_loss, test_loss = [], []
    correct = 0
    total = 0
    # Train the model
    for i, (images, labels) in enumerate(train_loader):  
        # Move tensors to the configured device
        images = images.to(device)
        labels = labels.to(device=device, dtype=torch.int64)
        
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss.append(loss.item())
    
    with torch.no_grad():
        
        # Predict the model
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device=device, dtype=torch.int64)
            outputs = model(images)
            lose = criterion(outputs, labels)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            test_loss.append(lose.item())

        print ("Epoch:", epoch + 1, ", Training Loss: ", np.mean(train_loss), ", Test loss: ", np.mean(test_loss))

# Set end time         
end = time.time()  
print('Accuracy is: {} %'.format(100 * correct / total))
print('It takes ' +  str(end - start)  + ' seconds to run Feedforward Neural Network algorithm on breast cancer dataset.')

#-----------------------------------------------------
# Adult Dataset
#-----------------------------------------------------

# Hyper-parameters 
input_size = 14
hidden_size = 10
num_classes = 2
num_epochs = 5
batch_size = 10
learning_rate = 0.01

# Read in data
data = pd.read_csv("adult.data",header = None)

# Drop patients' ID column
categorical_features = [1,3,5,6,7,8,9,13,14]

# Encode categorical variables
label_encoders = {}
for i in categorical_features:
    label_encoders[i] = LabelEncoder()
    data[i] = label_encoders[i].fit_transform(data[i])


# Make a class for breast adult dataset
class AdultData(Dataset):
    """Customized class that pre-processes adult dataset"""
    def __init__(self, csv_file):
        """
        Args:
            data: The audlt pandas dataframe 
        """
        self.main = data
        self.n = len(data)

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        features = self.main.iloc[idx, 0:14].values
        features = features.astype('double')
        features = np.array([[features]]) 
        features = torch.from_numpy(features)
        
            
        label = self.main.iloc[idx,14]

        sample = (features,label)


        return sample

# Get the dataset through the class
dataset = AdultData(data)

# Specify portion for splitting, shuffle data, set seed
percent_split = .2
shuffle_dataset = True
random_seed= 5

# Creating data indices for training and testing splits
dataset_size = len(dataset)
indices = list(range(dataset_size))
split = int(np.floor(percent_split * dataset_size))
if shuffle_dataset :
    np.random.seed(random_seed)
    np.random.shuffle(indices)
train_indices, val_indices = indices[split:], indices[:split]

# Creating PT data samplers and loaders
train_sampler = SubsetRandomSampler(train_indices)
test_sampler = SubsetRandomSampler(val_indices)

train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, 
                                           sampler=train_sampler)
test_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                                sampler=test_sampler)

# Convolutional neural network (two convolutional layers)
class ConvNet(nn.Module):
    def __init__(self, num_classes=10):
        super(ConvNet, self).__init__()
        torch.manual_seed(10) # For reproducibility
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.fc = nn.Linear(3*64, num_classes)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        return out

# Set model
model = ConvNet(num_classes).double().to(device)

# Set loss function and optimize algorithm
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)  

# Set start time
start = time.time()

for epoch in range(num_epochs):
    train_loss, test_loss = [], []
    correct = 0
    total = 0
    # Train the model
    for i, (images, labels) in enumerate(train_loader):  
        # Move tensors to the configured device
        #images = images.reshape(-1, 57).to(device)
        images = images.to(device)
        labels = labels.to(device=device, dtype=torch.int64)
        
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss.append(loss.item())
    
    with torch.no_grad():
        
        # Predict the model
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device=device, dtype=torch.int64)
            outputs = model(images)
            lose = criterion(outputs, labels)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            test_loss.append(lose.item())
            
        print ("Epoch:", epoch + 1, ", Training Loss: ", np.mean(train_loss), ", Test loss: ", np.mean(test_loss))

# Set end time         
end = time.time()  
print('Accuracy is: {} %'.format(100 * correct / total))
print('It takes ' +  str(end - start)  + ' seconds to run Feedforward Neural Network algorithm on adult dataset.')

#------------------------------------------------------------
# Madelon Dataset
#------------------------------------------------------------

# Hyper-parameters 
input_size = 500
hidden_size = 100
num_classes = 2
num_epochs = 5
batch_size = 10
learning_rate = 0.01

# Define a customized class for Madelyn data
class MadelonDataset(Dataset):
    """Customized class that pre-processes Madelyn dataset"""
    def __init__(self, x_file, y_file):
        """
        Args:
            csv_file (string): Name of dataset file
        """
        # Read in data
        self.main = pd.read_csv(x_file, header = None,sep = '\s+')
        self.lab = pd.read_csv(y_file, header = None,sep = '\s+')
    
    def __len__(self):
        return len(self.main)

    def __getitem__(self, idx):
        # Get the features and make it into tensor
        features = self.main.iloc[idx, 0:500].values
        features = features.astype('double')
        features = np.array([[features]])
        features = torch.from_numpy(features)
        # Get the label
        if self.lab.iloc[idx, 0] == -1:
            self.lab.iloc[idx, 0] = 0
            
        label = self.lab.iloc[idx, 0]
        # Combine features and label into a tuple
        sample = (features,label)
        return sample

train_dataset = MadelonDataset("madelon_train.data","madelon_train.labels")
test_dataset = MadelonDataset("madelon_valid.data","madelon_valid.labels")

# Data loader
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, 
                                           batch_size=batch_size, 
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset, 
                                          batch_size=batch_size, 
                                          shuffle=False)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') #CUDA is a GPU


# Convolutional neural network (two convolutional layers)
class ConvNet(nn.Module):
    def __init__(self, num_classes=10):
        super(ConvNet, self).__init__()
        torch.manual_seed(10) # For reproducibility
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.fc = nn.Linear(125*64, num_classes)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        return out

# Set model
model = ConvNet(num_classes).double().to(device)

# Set loss function and optimize algorithm
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)  

# Set start time
start = time.time()


for epoch in range(num_epochs):
    train_loss, test_loss = [], []
    correct = 0
    total = 0
    # Train the model
    for i, (x, y) in enumerate(train_loader):  
        # Move tensors to the configured device
        x = x.to(device)
        y = y.to(device=device, dtype=torch.int64)
        
        
        # Forward pass
        outputs = model(x)
        loss = criterion(outputs, y)
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss.append(loss.item())
    
    with torch.no_grad():
        
        # Predict the model
        for x, y in test_loader:
            x = x.to(device)
            y = y.to(device)
            outputs = model(x)
            lose = criterion(outputs, y)
            _, predicted = torch.max(outputs.data, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()
            test_loss.append(lose.item())

        print ("Epoch:", epoch + 1, ", Training Loss: ", np.mean(train_loss), ", Test loss: ", np.mean(test_loss))

# Set end time        
end = time.time()  
print('Accuracy is: {} %'.format(100 * correct / total))
print('It takes ' +  str(end - start)  + ' seconds to run Feedforward Neural Network algorithm on madelon dataset.')

#---------------------------------------------
# Parkinson Data
#---------------------------------------------

# Hyper-parameters 
input_size = 753
hidden_size = 500
num_classes = 2
num_epochs = 5
batch_size = 10
learning_rate = 0.01

# Define a customized class for Parkinson data
class ParkinsonDataset(Dataset):
    """Customized class that pre-processes parkinson dataset"""
    def __init__(self, csv_file):
        """
        Args:
            csv_file (string): Name of dataset file
        """
        # Read in data
        self.main = pd.read_csv(csv_file)
        
        # Standardize data
        #m = np.mean(self.landmarks_frame.iloc[:,0:57].values)
        #st = np.std(self.landmarks_frame.iloc[:,0:57].values)
        #self.landmarks_frame.iloc[:,0:57] = (self.landmarks_frame.iloc[:,0:57]-m)/st

    def __len__(self):
        return len(self.main)

    def __getitem__(self, idx):
        # Get the features and make it into tensor
        features = self.main.iloc[idx, 1:754].values
        features = np.array([[features]])
        features = torch.from_numpy(features)
        # Get the label
        label = self.main.iloc[idx, -1]
        # Combine features and label into a tuple
        sample = (features,label)
        return sample

# Get dataset
dataset = ParkinsonDataset("pd_speech_features.csv")

# Specify portion for splitting, shuffle data, set seed
test_split = .2
shuffle_dataset = True
random_seed= 5

# Creating data indices for training and testing splits
dataset_size = len(dataset)
indices = list(range(dataset_size))
split = int(np.floor(test_split * dataset_size))
if shuffle_dataset :
    np.random.seed(random_seed)
    np.random.shuffle(indices)
train_indices, val_indices = indices[split:], indices[:split]

# Creating data samplers and loaders

train_sampler = SubsetRandomSampler(train_indices)
test_sampler = SubsetRandomSampler(val_indices)

train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, 
                                           sampler=train_sampler)
test_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                                sampler=test_sampler)

# Convolutional neural network (two convolutional layers)
class ConvNet(nn.Module):
    def __init__(self, num_classes=10):
        super(ConvNet, self).__init__()
        torch.manual_seed(7) # For reproducibility
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.fc = nn.Linear(188*64, num_classes)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        return out

# Set model
model = ConvNet(num_classes).double().to(device)

# Set loss function and optimize algorithm
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)  

# Set start time
start = time.time()


for epoch in range(num_epochs):
    train_loss, test_loss = [], []
    correct = 0
    total = 0
    # Train the model
    for i, (x, y) in enumerate(train_loader):  
        # Move tensors to the configured device
        x = x.to(device)
        y = y.to(device)
        
        # Forward pass
        outputs = model(x)
        loss = criterion(outputs, y)
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss.append(loss.item())
    
    with torch.no_grad():
        
        # Predict the model
        for x, y in test_loader:
            x = x.to(device)
            y = y.to(device)
            outputs = model(x)
            lose = criterion(outputs, y)
            _, predicted = torch.max(outputs.data, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()
            test_loss.append(lose.item())

        print ("Epoch:", epoch + 1, ", Training Loss: ", np.mean(train_loss), ", Test loss: ", np.mean(test_loss))

# Set end time        
end = time.time()  
print('Accuracy is: {} %'.format(100 * correct / total))
print('It takes ' +  str(end - start)  + ' seconds to run Feedforward Neural Network algorithm on parkinson dataset.')

#---------------------------------------------
# SVHN Data
#---------------------------------------------

# Define a customized class for SVHN data
class SVHNDataset(Dataset):
    """Customized class that pre-processes SVHN dataset"""
    def __init__(self, file):
        """
	Args:
            file (string): Name of dataset file
        """
	# Read in data
        self.main = loadmat(file)
        self.x = self.main['X'].transpose().astype('float')
        self.y = self.main['y']

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        # Get the features and make it into tensor
        scaler = StandardScaler()
        self.x[idx] = [scaler.fit_transform(self.x[idx][i]) for i in range(0,3)]
        self.x[idx] = np.array(self.x[idx])
        features = torch.from_numpy(self.x[idx].flatten())
        # Get the label
        label = self.y[idx][0]
        if label == 10:
            label = 0
        # Combine features and label into a tuple
        sample = (features,label)
        return sample

training = SVHNDataset("train_32x32.mat")
testing = SVHNDataset("test_32x32.mat")

# Hyper-parameters
input_size = 3072
hidden_size = 3000
num_classes = 10
num_epochs = 30
batch_size = 1000
learning_rate = 0.001

train_loader = torch.utils.data.DataLoader(dataset=training,
                                           batch_size=batch_size,
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=testing,
                                          batch_size=batch_size,
                                          shuffle=False)

# Neural Network Class
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        torch.manual_seed(5)
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc4 = nn.Linear(hidden_size, num_classes)
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        out = self.relu(out)
        out = self.fc4(out)
        return out

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 

# Set model
model = NeuralNet(input_size, hidden_size, num_classes).double().to(device)

# Set loss function and optimize algorithm
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Set start time
start = time.time()


for epoch in range(num_epochs):
    train_loss, test_loss = [], []
    correct = 0
    total = 0
    # Train the model
    for i, (x, y) in enumerate(train_loader):
        # Move tensors to the configured device
        x = x.to(device)
        y = y.to(device=device, dtype=torch.int64)

    # Forward pass
        outputs = model(x)
        loss = criterion(outputs, y)

    # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss.append(loss.item())

    with torch.no_grad():

        # Predict the model
        for x, y in test_loader:
            x = x.to(device)
            y = y.to(device=device, dtype=torch.int64)
            outputs = model(x)
            lose = criterion(outputs, y)
            _, predicted = torch.max(outputs.data, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()
            test_loss.append(lose.item())

        print ("Epoch:", epoch + 1, ", Training Loss: ", np.mean(train_loss), ", Test loss: ", np.mean(test_loss))

# Set end time
# Set end time
end = time.time()
print('Accuracy is: {} %'.format(100 * correct / total))
print('It takes ' +  str(end - start)  + ' seconds to run Feedforward Neural Network algorithm on SVHN datasets.')

