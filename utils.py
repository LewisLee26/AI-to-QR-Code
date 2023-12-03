from collections import namedtuple

import torch
import torch.nn.functional as F

class FakeQuantOp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, num_bits=8, min_val=None, max_val=None):
        x = quantize_tensor(x,num_bits=num_bits, min_val=min_val, max_val=max_val)
        x = dequantize_tensor(x)
        return x

    @staticmethod
    def backward(ctx, grad_output):
        # straight through estimator
        return grad_output, None, None, None

# Get Min and max of x tensor, and stores it
def updateStats(x, stats, key):
    max_val, _ = torch.max(x, dim=1)
    min_val, _ = torch.min(x, dim=1)

    # add ema calculation

    if key not in stats:
        stats[key] = {"max": max_val.sum(), "min": min_val.sum(), "total": 1}
    else:
        stats[key]['max'] += max_val.sum().item()
        stats[key]['min'] += min_val.sum().item()
        stats[key]['total'] += 1
  
    weighting = 2.0 / (stats[key]['total']) + 1

    if 'ema_min' in stats[key]:
        stats[key]['ema_min'] = weighting*(min_val.mean().item()) + (1- weighting) * stats[key]['ema_min']
    else:
        stats[key]['ema_min'] = weighting*(min_val.mean().item())

    if 'ema_max' in stats[key]:
        stats[key]['ema_max'] = weighting*(max_val.mean().item()) + (1- weighting) * stats[key]['ema_max']
    else: 
        stats[key]['ema_max'] = weighting*(max_val.mean().item())

    stats[key]['min_val'] = stats[key]['min']/ stats[key]['total']
    stats[key]['max_val'] = stats[key]['max']/ stats[key]['total']
  
    return stats

# Reworked Forward Pass to access activation Stats through updateStats function
def gatherActivationStats(model, x, stats):

    stats = updateStats(x.clone().view(x.shape[0], -1), stats, 'conv1')
  
    x = F.relu(model.conv1(x))

    x = F.max_pool2d(x, 2, 2)
  
    stats = updateStats(x.clone().view(x.shape[0], -1), stats, 'conv2')
  
    x = F.relu(model.conv2(x))

    x = F.max_pool2d(x, 2, 2)

    x = x.view(-1, 4*4*50)
  
    stats = updateStats(x, stats, 'fc1')

    x = F.relu(model.fc1(x))
  
    stats = updateStats(x, stats, 'fc2')

    x = model.fc2(x)

    return stats

# Entry function to get stats of all functions.
def gatherStats(model, test_loader):
    device = 'cuda'
    
    model.eval()
    test_loss = 0
    correct = 0
    stats = {}
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            stats = gatherActivationStats(model, data, stats)
    
    final_stats = {}
    for key, value in stats.items():
      final_stats[key] = { "max" : value["max"] / value["total"], "min" : value["min"] / value["total"], "ema_min": value["ema_min"], "ema_max": value["ema_max"] }
    return final_stats

QTensor = namedtuple('QTensor', ['tensor', 'scale', 'zero_point'])

def calcScaleZeroPoint(min_val, max_val,num_bits=8):
    # Calc Scale and zero point of next 
    qmin = 0.
    qmax = 2.**num_bits - 1.

    scale = (max_val - min_val) / (qmax - qmin)

    initial_zero_point = qmin - min_val / scale
  
    zero_point = 0
    if initial_zero_point < qmin:
        zero_point = qmin
    elif initial_zero_point > qmax:
        zero_point = qmax
    else:
        zero_point = initial_zero_point

    zero_point = int(zero_point)

    return scale, zero_point

def calcScaleZeroPointSym(min_val, max_val,num_bits=8):
  
    # Calc Scale 
    max_val = max(abs(min_val), abs(max_val))
    qmin = 0.
    qmax = 2.**(num_bits-1) - 1.

    scale = max_val / qmax

    return scale, 0

def quantize_tensor(x, num_bits=8, min_val=None, max_val=None):
    
    if not min_val and not max_val: 
        min_val, max_val = x.min(), x.max()

    qmin = 0.
    qmax = 2.**num_bits - 1.

    scale, zero_point = calcScaleZeroPoint(min_val, max_val, num_bits)
    q_x = zero_point + x / scale
    q_x.clamp_(qmin, qmax).round_()
    q_x = q_x.round().byte()
    
    return QTensor(tensor=q_x, scale=scale, zero_point=zero_point)

def dequantize_tensor(q_x):
    return q_x.scale * (q_x.tensor.float() - q_x.zero_point)

def quantize_tensor_sym(x, num_bits=8, min_val=None, max_val=None):
    
    if not min_val and not max_val: 
        min_val, max_val = x.min(), x.max()

    max_val = max(abs(min_val), abs(max_val))
    qmin = 0.
    qmax = 2.**(num_bits-1) - 1.

    scale = max_val / qmax   

    q_x = x/scale

    q_x.clamp_(-qmax, qmax).round_()
    q_x = q_x.round()
    return QTensor(tensor=q_x, scale=scale, zero_point=0)

def dequantize_tensor_sym(q_x):
    return q_x.scale * (q_x.tensor.float())

def quantizeLayer(x, layer, stat, scale_x, zp_x, vis=False, axs=None, X=None, y=None, sym=False, num_bits=8):
    # for both conv and linear layers

    # cache old values
    W = layer.weight.data
    B = layer.bias.data

    # WEIGHTS SIMULATED QUANTISED

    # quantise weights, activations are already quantised
    if sym:
        w = quantize_tensor_sym(layer.weight.data,num_bits=num_bits) 
        b = quantize_tensor_sym(layer.bias.data,num_bits=num_bits)
    else:
        w = quantize_tensor(layer.weight.data, num_bits=num_bits) 
        b = quantize_tensor(layer.bias.data, num_bits=num_bits)

    layer.weight.data = w.tensor.float()
    layer.bias.data = b.tensor.float()

    ## END WEIGHTS QUANTISED SIMULATION

    # QUANTISED OP, USES SCALE AND ZERO POINT TO DO LAYER FORWARD PASS. (How does backprop change here ?)
    # This is Quantisation Arithmetic
    scale_w = w.scale
    zp_w = w.zero_point
    scale_b = b.scale
    zp_b = b.zero_point
  
    if sym:
        scale_next, zero_point_next = calcScaleZeroPointSym(min_val=stat['min'], max_val=stat['max'])
    else:
        scale_next, zero_point_next = calcScaleZeroPoint(min_val=stat['min'], max_val=stat['max'])

    # Preparing input by saturating range to num_bits range.
    if sym:
        X = x.float()
        layer.weight.data = ((scale_x * scale_w) / scale_next)*(layer.weight.data)
        layer.bias.data = (scale_b/scale_next)*(layer.bias.data)
    else:
        X = x.float() - zp_x
        layer.weight.data = ((scale_x * scale_w) / scale_next)*(layer.weight.data - zp_w)
        layer.bias.data = (scale_b/scale_next)*(layer.bias.data + zp_b)

    # All int computation
    if sym:  
        x = (layer(X)) 
    else:
        x = (layer(X)) + zero_point_next 
  
    # cast to int
    x.round_()

    # Perform relu too
    x = F.leaky_relu(x)

    # Reset weights for next forward pass
    layer.weight.data = W
    layer.bias.data = B
  
    return x, scale_next, zero_point_next


    
def quantAwareTrainingForward(model, x, stats, vis=False, axs=None, sym=False, num_bits=8, act_quant=False):
    conv1weight = model.conv1.weight.data
    model.conv1.weight.data = FakeQuantOp.apply(model.conv1.weight.data, num_bits)
    x = F.relu(model.conv1(x))

    with torch.no_grad():
        stats = updateStats(x.clone().view(x.shape[0], -1), stats, 'conv1')

    if act_quant:
        x = FakeQuantOp.apply(x, num_bits, stats['conv1']['ema_min'], stats['conv1']['ema_max'])

    x = F.max_pool2d(x, 2, 2)

    conv2weight = model.conv2.weight.data
    model.conv2.weight.data = FakeQuantOp.apply(model.conv2.weight.data, num_bits)
    x = F.relu(model.conv2(x))

    with torch.no_grad():
        stats = updateStats(x.clone().view(x.shape[0], -1), stats, 'conv2')
    
    if act_quant:
        x = FakeQuantOp.apply(x, num_bits, stats['conv2']['ema_min'], stats['conv2']['ema_max'])


    x = F.max_pool2d(x, 2, 2)

    x = x.view(-1, 4*4*8)

    fc1weight = model.fc1.weight.data
    model.fc1.weight.data = FakeQuantOp.apply(model.fc1.weight.data, num_bits)
    x = F.relu(model.fc1(x))

    with torch.no_grad():
        stats = updateStats(x.clone().view(x.shape[0], -1), stats, 'fc1')

    if act_quant:
        x = FakeQuantOp.apply(x, num_bits, stats['fc1']['ema_min'], stats['fc1']['ema_max'])

    # x = model.fc2(x)
  
    with torch.no_grad():
        stats = updateStats(x.clone().view(x.shape[0], -1), stats, 'fc2')

    return F.log_softmax(x, dim=1), conv1weight, conv2weight, fc1weight, stats

def trainQuantAware(args, model, device, train_loader, optimizer, epoch, stats, act_quant=False, num_bits=4):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output, conv1weight, conv2weight, fc1weight, stats = quantAwareTrainingForward(model, data, stats, num_bits=num_bits, act_quant=act_quant)

        model.conv1.weight.data = conv1weight
        model.conv2.weight.data = conv2weight
        model.fc1.weight.data = fc1weight

        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()

        if batch_idx % args["log_interval"] == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
    return stats

def testQuantAware(args, model, device, test_loader, stats, act_quant, num_bits=4):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output, conv1weight, conv2weight, fc1weight, _ = quantAwareTrainingForward(model, data, stats, num_bits=num_bits, act_quant=act_quant)
            
            model.conv1.weight.data = conv1weight
            model.conv2.weight.data = conv2weight
            model.fc1.weight.data = fc1weight

            test_loss += F.nll_loss(output, target, reduction='sum').item() # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))