
import torch
import torch.optim as optim
from torchvision import datasets, transforms
from model import Net
from utils import trainQuantAware, testQuantAware

def mainQuantAware(mnist=True):
 
    batch_size = 64
    test_batch_size = 64
    epochs = 10
    lr = 0.01
    momentum = 0.5
    seed = 1
    log_interval = 500
    save_model = True
    no_cuda = False
    
    use_cuda = not no_cuda and torch.cuda.is_available()

    torch.manual_seed(seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    if mnist:
      train_loader = torch.utils.data.DataLoader(
          datasets.MNIST('../data', train=True, download=True,
                        transform=transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize((0.1307,), (0.3081,))
                        ])),
          batch_size=batch_size, shuffle=True, **kwargs)
      
      test_loader = torch.utils.data.DataLoader(
          datasets.MNIST('../data', train=False, transform=transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize((0.1307,), (0.3081,))
                        ])),
          batch_size=test_batch_size, shuffle=True, **kwargs)
    else:
        transform = transforms.Compose(
            [transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        trainset = datasets.CIFAR10(root='./dataCifar', train=True,
                                              download=True, transform=transform)
        train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                                shuffle=True, num_workers=2)

        testset = datasets.CIFAR10(root='./dataCifar', train=False,
                                            download=True, transform=transform)
        test_loader = torch.utils.data.DataLoader(testset, batch_size=test_batch_size,
                                              shuffle=False, num_workers=2)
          
  
    model = Net(mnist=mnist).to(device)
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    args = {}
    args["log_interval"] = log_interval
    epochs = 10
    num_bits=3
    stats = {}
    for epoch in range(1, epochs + 1):
        if epoch > 5:
          act_quant = True 
        else:
          act_quant = False

        stats = trainQuantAware(args, model, device, train_loader, optimizer, epoch, stats, act_quant, num_bits=num_bits)
        testQuantAware(args, model, device, test_loader, stats, act_quant, num_bits=num_bits)

    if (save_model):
        torch.save(model.state_dict(),"models/mnist_cnn.pth")

    return model, stats

if __name__ == '__main__':
    model, old_stats = mainQuantAware()

