# visualize current GPU usages in your server
!nvidia-smi 

# set gpu by number 
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # setting gpu number

# load packages
!pip install torch
!pip install numpy
import torch
import numpy as np

# print the version of PyTorch
print(torch.__version__)

np_array_1 = np.array([1, 2, 3, 4])
np_array_2 = np.array([5, 6, 7, 8])
torch_tensor_1 = torch.tensor([1, 2, 3, 4])
torch_tensor_2 = torch.tensor([5 ,6 ,7, 8])

print (np_array_1)
print (np_array_2)
print (torch_tensor_1)
print (torch_tensor_2)

# numpy
print (np_array_1.shape)

# torch
print (torch_tensor_1.shape)

print (torch_tensor_1.size()) # size() and shape operation is identical in torch

# numpy
np_concate = np.concatenate([np_array_1, np_array_2], axis=0)
print ('----numpy----')
print (np_concate)

# torch
torch_concate = torch.cat([torch_tensor_1, torch_tensor_2], dim=0)
print ('----torch----')
print (torch_concate)

# dim=1로 하려면 dimension을 증가시켜줘야한다
torch_concate_dim1= torch.cat([torch_tensor_1[:,None], torch_tensor_2[:,None]], dim=1)
print (torch_concate_dim1)
# torch_tensor_1 = torch_tensor_1.reshape(1,4)
# torch_tensor_2 = torch_tensor_2.reshape(1,4)

# numpy
np_reshaped = np_concate.reshape(4,2)
print ('----numpy----')
print (np_reshaped)
print (np_reshaped.shape)

# torch
torch_reshaped = torch_concate.view(4, 2)
# torch에서도 reshape 된다고 하심
print ('----torch----')
print (torch_reshaped)
print (torch_reshaped.shape)

x = np.array([1, 2, 3])
x_repeat = x.repeat(2) # numpy의 repeat은  11 22 33 이렇게 된다

print ('----numpy----')
print (x)
print (x_repeat)

x = torch.tensor([1, 2, 3])
x_repeat = x.repeat(2) # torch의 repeat은 123 123

print ('----torch----')
print (x)
print (x_repeat)

# To obtain the same result with np.repeat (will skip explanation: you should be proficient with reshaping operations)
print('----obtain the same result-----')
x_repeat = x.view(3, 1)
print (x_repeat)

x_repeat = x_repeat.repeat(1, 2)
print (x_repeat)

x_repeat = x_repeat.view(-1)
print (x_repeat)

# similar manipulation operation: stack & repeat
x = torch.tensor([1, 2, 3])
x_repeat = x.repeat(4) # 가로로 repeat하지만
x_stack = torch.stack([x, x, x, x]) # stack은 세로로 쌓는다

print (x_repeat)
print (x_stack)
print (x_repeat.view(4, 3)) # reshape x

print(torch.cuda.is_available())  # Is GPU accessible?

a = torch.ones(3)
b = torch.randn(100, 50, 3)

print(b.shape)

print(a.device)
print(b.device)

# tensor.device : tensor가 cpu에 located되어있다
# 우리는 이걸 GPU에 올리고 싶다

c = a + b

print(c.device)

# c도 cpu에 located되어있다

# upload a and b to GPU
a = a.to('cuda')
b = b.to('cuda')

print(a.device)
print(b.device)

x = torch.ones(2, 2, requires_grad=True)
print(x)

y = x + 2
print(y)

z = y * y * 3
print(z)

out = z.mean()
print(out)

y.retain_grad()

z.retain_grad()
out.backward()

with torch.no_grad():
    x = torch.ones(2, 2, requires_grad=True)
    y = x + 2
    z = y * y * 3
    out = z.mean()
    
    # out.backward() ## ERROR!!!!: we used torch.no_grad()!!

# since it does not save any gradient
# therefore it's much faster

import torch.nn as nn

X = torch.tensor([[1., 2., 3.], [4., 5., 6.]])

print (X)
print (X.shape) 

# input dim 3, output dim 1
linear_fn = nn.Linear(3, 1)

Y = linear_fn(X)
print(Y)
print(Y.shape)

# 이 값들이 어떻게 나온거지???????

class Model(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim):
        super(Model, self).__init__()
        self.linear_1 = nn.Linear(input_dim, hidden_dim) # input_dim -> hidden_dim FC_layer
        self.linear_2 = nn.Linear(hidden_dim, output_dim) # hidden_dim -> output_dim FC_layer
        self.relu = nn.ReLU() # Activation function
        
    def forward(self, x):
        x = self.linear_1(x)
        x = self.relu(x) # Activation function
        x = self.linear_2(x)
        return x
    
    nn.Sigmoid
nn.ReLU
nn.LeakyReLU
nn.Tanh;

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.utils.data import DataLoader

import torchvision
import torchvision.transforms as transforms

# MNIST dataset 
# torch에서 제공한다
train_dataset = torchvision.datasets.MNIST(root='./', train=True, transform=transforms.ToTensor(), download=True)
test_dataset = torchvision.datasets.MNIST(root='./', train=False, transform=transforms.ToTensor())

# Data loader
# mini batch size
train_loader = DataLoader(dataset=train_dataset, batch_size=128, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=100, shuffle=False)

# Define model class
# This model has one hidden layer

# multinomial이니까 output_size가 1보다 크겠지?
class Multinomial_logistic_regression(nn.Module):
    def __init__(self, input_size, output_size):
        super(Multinomial_logistic_regression, self).__init__()
        self.fc = nn.Linear(input_size, output_size)
        
    def forward(self, x):
        out = self.fc(x)
        return out
    
    # Generate model
model = Multinomial_logistic_regression(784, 10)  # init(784, 10)
# input dim: 784  / output dim: 10
# 28 x 28

# Optimizer define
optimizer = torch.optim.SGD(model.parameters(), lr=0.05) 
# optimizer = torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.9)
# optimizer = torch.optim.Adam(model.parameters(), lr=0.05)

# Loss function define (we use cross-entropy)
loss_fn = nn.CrossEntropyLoss()

#Train the model
total_step = len(train_loader)

for epoch in range(10):
    for i, (images, labels) in enumerate(train_loader):  # mini batch for loop
        # upload to gpu
        images = images.reshape(-1, 28*28).to('cuda') # FLATTEN 하는거다
        labels = labels.to('cuda')
        
        # Forward
        outputs = model(images)  # forwardI(images): get prediction
        loss = loss_fn(outputs, labels)  # calculate the loss (crossentropy loss) with ground truth & prediction value
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()  # automatic gradient calculation (autograd)
        optimizer.step()  # update model parameter with requires_grad=True 
        
        if (i+1) % 100 == 0:
            print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
                   .format(epoch+1, 10, i+1, total_step, loss.item()))
            
            # Test the model
# In test phase, we don't need to compute gradients (for memory efficiency)
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = images.reshape(-1, 28*28).to('cuda')
        labels = labels.to('cuda')
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)  # classificatoin model -> get the label prediction of top 1 
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the 10000 test images: {} %'.format(100 * correct / total))
    
    # New model with multi layer
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size) 
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()  # sigmoid activation function (you can customize)
    
    def forward(self, x):
        out = self.fc1(x)
        out = self.sigmoid(out)
        out = self.fc2(out)
        out = self.sigmoid(out)
        out = self.fc3(out)
        return out
    
    # Generate model
model = NeuralNet(784, 20, 10)  # init(784, 20, 10)
# input dim: 784  / hidden dim: 20  / output dim: 10

# Upload model to GPU
model = model.to('cuda')

# Loss function define (we use cross-entropy)
loss_fn = nn.CrossEntropyLoss()

# Define optimizer
# optimizer = torch.optim.SGD(model.parameters(), lr=0.05) 
optimizer = torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.9)
# optimizer = torch.optim.Adam(model.parameters(), lr=0.05)

# Train the model
total_step = len(train_loader)

for epoch in range(10):
    for i, (images, labels) in enumerate(train_loader):  # mini batch for loop
        # upload to gpu
        images = images.reshape(-1, 28*28).to('cuda')
        labels = labels.to('cuda')
        
        # Forward
        outputs = model(images)  # forwardI(images): get prediction
        loss = loss_fn(outputs, labels)  # calculate the loss (crossentropy loss) with ground truth & prediction value
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()  # automatic gradient calculation (autograd)
        optimizer.step()  # update model parameter with requires_grad=True 
        
        if (i+1) % 100 == 0:
            print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
                   .format(epoch+1, 10, i+1, total_step, loss.item()))
            
            # Test the model
# In test phase, we don't need to compute gradients (for memory efficiency)
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = images.reshape(-1, 28*28).to('cuda')
        labels = labels.to('cuda')
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)  # classificatoin model -> get the label prediction of top 1 
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the 10000 test images: {} %'.format(100 * correct / total))
    
    ## 반복
    
    for epoch in range(10):
        for i, (images, labels) in enumerate(train_loader):  # mini batch for loop
        # upload to gpu
        images = images.reshape(-1, 28*28).to('cuda') # FLATTEN 하는거다
        labels = labels.to('cuda')
        
        # Forward
        outputs = model(images)  # forwardI(images): get prediction
        loss = loss_fn(outputs, labels)  # calculate the loss (crossentropy loss) with ground truth & prediction value
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()  # automatic gradient calculation (autograd)
        optimizer.step()  # update model parameter with requires_grad=True 
        
        if (i+1) % 100 == 0:
            print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
                   .format(epoch+1, 10, i+1, total_step, loss.item()))
            
            # Test the model
# In test phase, we don't need to compute gradients (for memory efficiency)
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = images.reshape(-1, 28*28).to('cuda')
        labels = labels.to('cuda')
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)  # classificatoin model -> get the label prediction of top 1 
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the 10000 test images: {} %'.format(100 * correct / total))
    
    for epoch in range(10):
        for i, (images, labels) in enumerate(train_loader):  # mini batch for loop
        # upload to gpu
        images = images.reshape(-1, 28*28).to('cuda') # FLATTEN 하는거다
        labels = labels.to('cuda')
        
        # Forward
        outputs = model(images)  # forwardI(images): get prediction
        loss = loss_fn(outputs, labels)  # calculate the loss (crossentropy loss) with ground truth & prediction value
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()  # automatic gradient calculation (autograd)
        optimizer.step()  # update model parameter with requires_grad=True 
        
        if (i+1) % 100 == 0:
            print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
                   .format(epoch+1, 10, i+1, total_step, loss.item()))
            
            # Test the model
# In test phase, we don't need to compute gradients (for memory efficiency)
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = images.reshape(-1, 28*28).to('cuda')
        labels = labels.to('cuda')
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)  # classificatoin model -> get the label prediction of top 1 
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the 10000 test images: {} %'.format(100 * correct / total))
    
    for epoch in range(10):
        for i, (images, labels) in enumerate(train_loader):  # mini batch for loop
        # upload to gpu
        images = images.reshape(-1, 28*28).to('cuda') # FLATTEN 하는거다
        labels = labels.to('cuda')
        
        # Forward
        outputs = model(images)  # forwardI(images): get prediction
        loss = loss_fn(outputs, labels)  # calculate the loss (crossentropy loss) with ground truth & prediction value
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()  # automatic gradient calculation (autograd)
        optimizer.step()  # update model parameter with requires_grad=True 
        
        if (i+1) % 100 == 0:
            print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
                   .format(epoch+1, 10, i+1, total_step, loss.item()))
            
            # Test the model
# In test phase, we don't need to compute gradients (for memory efficiency)
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = images.reshape(-1, 28*28).to('cuda')
        labels = labels.to('cuda')
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)  # classificatoin model -> get the label prediction of top 1 
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the 10000 test images: {} %'.format(100 * correct / total))
    
    with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = images.reshape(-1, 28*28).to('cuda')
        labels = labels.to('cuda')
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)  # classificatoin model -> get the label prediction of top 1 
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the 10000 test images: {} %'.format(100 * correct / total))