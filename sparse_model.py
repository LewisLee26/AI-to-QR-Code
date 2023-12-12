import torch 
import torch.nn as nn
import torch.nn.utils.prune as prune
import gzip
import torch.optim as optim
import torch.onnx

from model import Net

from torchvision import datasets, transforms

from utils import trainQuantAware, testQuantAware
if __name__ == '__main__':

    test_batch_size = 64
    batch_size = 64
    no_cuda = False

    use_cuda = not no_cuda and torch.cuda.is_available()

    device = torch.device("cuda" if use_cuda else "cpu")
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
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

    model = Net(mnist=True).to(device)
    model.load_state_dict(torch.load("models\mnist_cnn.pth"))

    args = {}
    args["log_interval"] = 500
    epochs = 5
    num_bits=3
    stats = {}
    act_quant = True 
    lr = 0.01
    momentum = 0.5
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)

    testQuantAware(args, model, device, test_loader, stats, act_quant, num_bits=num_bits)

    count_masked_parameters = 0
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            prune.l1_unstructured(module, name='weight', amount=0.85)
            # use the mask, do not remove the weights 
            mask = module.weight_mask  # Get the pruning mask
            count_masked_parameters += torch.sum(mask != 0).item()  # Count non-zero elements in the mask
        elif isinstance(module, nn.Linear):
            prune.l1_unstructured(module, name='weight', amount=0.9)
            mask = module.weight_mask  # Get the pruning mask
            count_masked_parameters += torch.sum(mask != 0).item()  # Count non-zero elements in the mask
    
    print(f'The model has {count_masked_parameters} masked parameters')


    testQuantAware(args, model, device, test_loader, stats, act_quant, num_bits=num_bits)

    for epoch in range(1, epochs + 1):
        if epoch > 2:
          act_quant = True 
        else:
          act_quant = False

        stats = trainQuantAware(args, model, device, train_loader, optimizer, epoch, stats, act_quant, num_bits=num_bits)
        testQuantAware(args, model, device, test_loader, stats, act_quant, num_bits=num_bits)

    torch.save(model, r"models/mnist_cnn_pruned.pth")

    torch.onnx.export(model, torch.randn(1, 1, 28, 28).to(device), 'models/mnist_cnn_pruned.onnx')

    with gzip.open('models/mnist_cnn_pruned.gz', 'wb') as f:
        torch.onnx.export(model, torch.randn(1, 1, 28, 28).to(device), f)
