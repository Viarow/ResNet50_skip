import torch
import torchvision
import torchvision.transforms as transforms
import os
import torch.nn as nn
import torch.optim as optim
import argparse
from model import ResNet50


def train(args):

    transform = transforms.Compose(
        [transforms.Resize(224),
         transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=False, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size,
                                              shuffle=True, num_workers=2)
    print('Dataset loaded.')

    model = ResNet50(num_classes=10)
    print('Model built.')


    if torch.cuda.device_count()>1:
        model = nn.DataParallel(model).cuda()
        print('Model data parallel to cuda.')
    else:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model = model.to(device)



    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum,
                          weight_decay=args.weight_decay)

    model.train()
    epoch_num = args.epoch
    for epoch in range(epoch_num):

        running_loss = 0.0
        for i,data in enumerate(trainloader,0):

            inputs, labels = data
            inputs, labels = inputs.cuda(), labels.cuda()

            optimizer.zero_grad()

            outputs = model(inputs)
            #flops = model.flops
            #compute_cost = args.alpha * flops
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 2000 == 1999:
                #print("epoch-%d sample-%d compute_cost: %.3f" % (epoch+1, i+1, compute_cost))
                print("epoch-%d sample-%d running_loss: %.3f" % (epoch + 1, i + 1, running_loss))
                running_loss = 0.0


def parse_args():

    parser = argparse.ArgumentParser()
    parser.add_argument('-epoch', type=int, default=2)
    parser.add_argument('-momentum', type=float, default=0.9)
    parser.add_argument('-lr', type=float, default=5e-2)
    parser.add_argument('-weight_decay', type=float, default=1e-4)
    parser.add_argument('-batch_size', type=int, default=4)
    parser.add_argument('-alpha', type=float, default=2e-4)
    args = parser.parse_args()

    return args


if __name__ =='__main__':

    args = parse_args()
    train(args)



