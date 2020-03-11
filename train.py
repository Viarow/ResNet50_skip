import torch
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
import argparse
from model import ResNet50

FLOP_list = []

def compute_flops(module, input, output):

    c_in = input[0].shape[1]
    k_size = 3
    h, w = input[0].shape[2], input[0].shape[3]
    batch_size = output[0].shape[0]
    flops = 0

    for k in range(0, batch_size):
        c_out = output[0][k].cpu().sum()
        flops += ( 2*c_in*k_size*k_size - 1 )*h*w*c_out
        FLOP_list.append(flops)


def register_layers(model):
    handles = []
    for idx, m in enumerate(model.named_modules()):
        module_name = m[0]
        if module_name.split('.')[-1] == 'channel_gate':
            handle = m[1].register_forward_hook(compute_flops)
            handles.append(handle)
    return handles


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

    handles = register_layers(model)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum,
                          weight_decay=args.weight_decay)

    model.train()
    epoch_num = args.epoch
    for epoch in range(epoch_num):

        running_loss = 0.0
        for i,data in enumerate(tqdm(trainloader)):

            inputs, labels = data
            inputs, labels = inputs.cuda(), labels.cuda()

            optimizer.zero_grad()
            outputs = model(inputs)

            for handle in handles:
                handle.remove()
            handles = []
            compute_cost = sum(FLOP_list) * args.alpha

            loss = criterion(outputs, labels) + torch.Tensor([compute_cost]).cuda()
            #loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if i % args.checkpoint == (args.checkpoint-1) :
                print("epoch-%d sample-%d running_loss: %.3f" % (epoch + 1, i + 1, running_loss/args.checkpoint))
                running_loss = 0.0



def parse_args():

    parser = argparse.ArgumentParser()
    parser.add_argument('-epoch', type=int, default=2)
    parser.add_argument('-momentum', type=float, default=0.9)
    parser.add_argument('-lr', type=float, default=5e-2)
    parser.add_argument('-weight_decay', type=float, default=1e-4)
    parser.add_argument('-batch_size', type=int, default=4)
    parser.add_argument('-alpha', type=float, default=5e-12)
    parser.add_argument('-checkpoint', type=int, default=50)
    args = parser.parse_args()

    return args


if __name__ =='__main__':

    args = parse_args()
    train(args)



