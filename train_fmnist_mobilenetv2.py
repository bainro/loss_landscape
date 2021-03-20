#from sklearn.model_selection import train_test_split
from torchvision import datasets, transforms, models
from torch.utils.data import Dataset, TensorDataset
import torch.nn.functional as F
from torch import nn, optim
import torchvision
import numpy as np
import sys
import torch


assert(len(sys.argv) > 1)
save_path = sys.argv[1]

n_epochs = 15
log_interval = 100
use_cuda = torch.cuda.is_available()
if use_cuda:
    torch.cuda.empty_cache()
    torch.cuda.set_device(0)

preprocess_train = transforms.Compose([
    transforms.RandomAffine(10, translate=(0.1,0.1)),
    transforms.RandomResizedCrop(size=28, scale=(0.9,1.1), ratio=(0.95, 1.05)),
    transforms.Resize(256),
    transforms.Grayscale(num_output_channels=3),
    transforms.CenterCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

preprocess_test = transforms.Compose([
    transforms.Resize(224),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

trainset = torchvision.datasets.FashionMNIST(root="fashion", train=True, download=True, transform=preprocess_test)
testset = torchvision.datasets.FashionMNIST(root="fashion", train=False, download=True, transform=preprocess_test)

train_loader = torch.utils.data.DataLoader(trainset, batch_size=256, shuffle=True, num_workers=2)
test_loader = torch.utils.data.DataLoader(testset, batch_size=256, shuffle=False, num_workers=2)

net = models.mobilenet_v2(num_classes=10, width_mult=0.125)
if use_cuda:
    net.cuda()
optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
    factor=0.1, patience=5, threshold=0.0001, threshold_mode='abs')
criterion = nn.CrossEntropyLoss().cuda()

def train(epoch):
  net.train()
  for batch_idx, (data, target) in enumerate(train_loader):
    if use_cuda:
        data, target = data.cuda(), target.cuda()
    optimizer.zero_grad()
    output = net(data)
    loss = criterion(output, target)
    loss.backward()
    optimizer.step()
    if batch_idx % log_interval == 0:
      scheduler.step(epoch)
      print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
        epoch, batch_idx * len(data), len(train_loader.dataset),
        100. * batch_idx / len(train_loader), loss.item()))


def test():
  net.eval()
  test_loss = 0
  correct = 0
  with torch.no_grad():
    for data, target in test_loader:
      if use_cuda:
        data, target = data.cuda(), target.cuda()
      output = net(data)
      test_loss += criterion(output, target).item()
      pred = output.data.max(1, keepdim=True)[1]
      correct += pred.eq(target.data.view_as(pred)).sum()
  test_loss /= len(test_loader.dataset)
  print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
    test_loss, correct, len(test_loader.dataset),
    100. * correct / len(test_loader.dataset)))

def save_model(net, path):
    torch.save(net.state_dict(), path)
    #torch.save(optimizer.state_dict(), 'results/_optimizer_.pth')

# save_model(net, "results/init_mobilev2_2.pth")
for epoch in range(1, n_epochs + 1):
  train(epoch)
  test()

save_model(net, "models/" + save_path + ".pth")
