#from sklearn.model_selection import train_test_split
from torchvision import datasets, transforms, models
from torch.utils.data import Dataset, TensorDataset
import torch.nn.functional as F
from torch import nn, optim
import torchvision
import dataloader
import numpy as np
import torch

n_epochs = 200
log_interval = 100
use_cuda = torch.cuda.is_available()
if use_cuda:
    torch.cuda.set_device(1)
    #torch.cuda.empty_cache()

train_trans = transforms.Compose([
    transforms.RandomAffine(10, translate=(0.1,0.1)),
    transforms.RandomCrop(32, padding=4),
    transforms.Resize(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
test_trans = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))])


train_loader, test_loader = dataloader.load_dataset(dataset='cifar10', datapath='cifar10/data', batch_size=128, \
                                                    threads=2, raw_data=False, data_split=1, split_idx=0, \
                                                    trainloader_path="", testloader_path="", eval_count=None, ts=[train_trans, test_trans])

net = models.mobilenet_v2(num_classes=10, width_mult=1.0)
if use_cuda:
    net.cuda()
optimizer = torch.optim.Adam(net.parameters(), lr=0.02)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
    factor=0.1, patience=10, threshold=0.0001, threshold_mode='abs')
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

save_model(net, "results/init_mobilev2_1w_cifar10.pth")
for epoch in range(1, n_epochs + 1):
  train(epoch)
  test()
  if epoch % 10 == 0:
    save_model(net, "results/mobilev2_1w_cifar10.pth")