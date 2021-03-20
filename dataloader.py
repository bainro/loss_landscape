import torch
import torchvision
from torchvision import transforms
import os
import numpy as np
import argparse

def get_relative_path(file):
    script_dir = os.path.dirname(__file__)  # <-- absolute dir the script is in
    return os.path.join(script_dir, file)


def load_dataset(dataset='cifar10', datapath='cifar10/data', batch_size=128, \
                 threads=2, raw_data=False, data_split=1, split_idx=0, \
                 trainloader_path="", testloader_path="", eval_count=None, ts=[]):
    """
    Setup dataloader. The data is not randomly cropped as in training because of
    we want to esimate the loss value with a fixed dataset.

    Args:
        raw_data: raw images, no data preprocessing
        data_split: the number of splits for the training dataloader
        split_idx: the index for the split of the dataloader, starting at 0

    Returns:
        train_loader, test_loader
    """

    # use specific dataloaders
    if trainloader_path and testloader_path:
        assert os.path.exists(trainloader_path), 'trainloader does not exist'
        assert os.path.exists(testloader_path), 'testloader does not exist'
        train_loader = torch.load(trainloader_path)
        test_loader = torch.load(testloader_path)
        return train_loader, test_loader

    assert split_idx < data_split, 'the index of data partition should be smaller than the total number of split'

    if dataset == 'cifar100':
        preprocess = transforms.Compose([
            transforms.ToTensor()
        ])

        trainset = torchvision.datasets.CIFAR100(root="cifar10", train=True, download=True, transform=preprocess)
        testset = torchvision.datasets.CIFAR100(root="cifar10", train=False, download=True, transform=preprocess)

        train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=threads)
        test_loader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False, num_workers=threads)

        if eval_count != None:
            _size = len(testset) - eval_count
            _, testset = torch.utils.data.random_split(testset, [_size, eval_count])
            
            print("# of eval ex's:", len(testset))
            test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                                  shuffle=False, num_workers=threads)

    elif dataset == 'cifar10':
        normalize = transforms.Normalize(mean=[x/255.0 for x in [125.3, 123.0, 113.9]],
                                         std=[x/255.0 for x in [63.0, 62.1, 66.7]])

        data_folder = get_relative_path(datapath)
        
        if raw_data:
            transform = transforms.Compose([
                transforms.ToTensor()
            ])
        else:
            transform = transforms.Compose([
                transforms.ToTensor(),
                normalize,
            ])

        if len(ts) != 2:
            trainset = torchvision.datasets.CIFAR10(root=data_folder, train=True,
                                                download=True, transform=transform)
        else:
            trainset = torchvision.datasets.CIFAR10(root=data_folder, train=True,
                                                download=True, transform=ts[0])
        # If data_split>1, then randomly select a subset of the data. E.g., if datasplit=3, then
        # randomly choose 1/3 of the data.
        if data_split > 1:
            indices = torch.tensor(np.arange(len(trainset)))
            data_num = len(trainset) // data_split # the number of data in a chunk of the split

            # Randomly sample indices. Use seed=0 in the generator to make this reproducible
            state = np.random.get_state()
            np.random.seed(0)
            indices = np.random.choice(indices, data_num, replace=False)
            np.random.set_state(state)

            train_sampler = torch.utils.data.sampler.SubsetRandomSampler(indices)
            train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                                       sampler=train_sampler,
                                                       shuffle=False, num_workers=threads)
        else:
            kwargs = {'num_workers': 2, 'pin_memory': True}
            train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                                      shuffle=False, **kwargs)
        if len(ts) != 2:
            testset = torchvision.datasets.CIFAR10(root=data_folder, train=False,
                                                download=False, transform=transform)
        else:
            testset = torchvision.datasets.CIFAR10(root=data_folder, train=False,
                                            download=False, transform=ts[1])
        
        #print(eval_count)
        if eval_count != None:
            _size = len(testset) - eval_count
            _, test_dataset = torch.utils.data.random_split(testset, [_size, eval_count])
            
            print("# of eval ex's:", len(test_dataset))
            test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size,
                                                  shuffle=False, num_workers=threads)
        else:
            test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                                  shuffle=False, num_workers=threads)

    elif dataset == "mnist":
        preprocess_train = transforms.Compose([
            transforms.RandomAffine(10, translate=(0.1,0.1)),
            transforms.RandomResizedCrop(size=28, scale=(0.9,1.1), ratio=(0.95, 1.05)),
            transforms.Resize(256),
            transforms.Grayscale(num_output_channels=3),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        preprocess_test = transforms.Compose([
            transforms.Resize(224),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        trainset = torchvision.datasets.MNIST(root="mnist", train=True, download=True, transform=preprocess_train)
        testset = torchvision.datasets.MNIST(root="mnist", train=False, download=True, transform=preprocess_test)

        if eval_count != None:
            _size = len(testset) - eval_count
            _, testset = torch.utils.data.random_split(testset, [_size, eval_count])
            
            print("# of eval ex's:", len(testset))
            test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                                  shuffle=False, num_workers=threads)

        train_loader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True, num_workers=2)
        test_loader = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=False, num_workers=2)

    elif dataset == "fmnist":
        preprocess_train = transforms.Compose([
            transforms.ToTensor()
        ])

        preprocess_test = transforms.Compose([
            transforms.ToTensor(),
        ])

        trainset = torchvision.datasets.FashionMNIST(root="fashion", train=True, download=True, transform=preprocess_train)
        testset = torchvision.datasets.FashionMNIST(root="fashion", train=False, download=True, transform=preprocess_test)

        if eval_count != None:
            _size = len(testset) - eval_count
            _, testset = torch.utils.data.random_split(testset, [_size, eval_count])
            
            print("# of eval ex's:", len(testset))
            test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                                  shuffle=False, num_workers=threads)

        train_loader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True, num_workers=2)
        test_loader = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=False, num_workers=2)

    return train_loader, test_loader


###############################################################
####                        MAIN
###############################################################

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
    parser.add_argument('--mpi', '-m', action='store_true', help='use mpi')
    parser.add_argument('--cuda', '-c', action='store_true', help='use cuda')
    parser.add_argument('--threads', default=2, type=int, help='number of threads')
    parser.add_argument('--batch_size', default=128, type=int, help='minibatch size')
    parser.add_argument('--dataset', default='cifar10', help='cifar10 | imagenet')
    parser.add_argument('--datapath', default='cifar10/data', metavar='DIR', help='path to the dataset')
    parser.add_argument('--raw_data', action='store_true', default=False, help='do not normalize data')
    parser.add_argument('--data_split', default=1, type=int, help='the number of splits for the dataloader')
    parser.add_argument('--split_idx', default=0, type=int, help='the index of data splits for the dataloader')
    parser.add_argument('--trainloader', default='', help='path to the dataloader with random labels')
    parser.add_argument('--testloader', default='', help='path to the testloader with random labels')

    args = parser.parse_args()

    trainloader, testloader = load_dataset(args.dataset, args.datapath,
                                args.batch_size, args.threads, args.raw_data,
                                args.data_split, args.split_idx,
                                args.trainloader, args.testloader)

    print('num of batches: %d' % len(trainloader))
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        print('batch_idx: %d   batch_size: %d'%(batch_idx, len(inputs)))
