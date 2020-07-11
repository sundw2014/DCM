"""
    PyTorch training code for Wide Residual Networks:
    http://arxiv.org/abs/1605.07146

    The code reproduces *exactly* it's lua version:
    https://github.com/szagoruyko/wide-residual-networks

    2016 Sergey Zagoruyko
"""

import argparse
import os
import json
import numpy as np
from tqdm import tqdm
import torch
from torch.optim import SGD
import torch.utils.data
import torchvision.transforms as T
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torchnet as tnt
from torchnet.engine import Engine
from utils import cast, data_parallel, print_tensor_dict
from torch.backends import cudnn
from resnet import resnet
from KL_divergence import KL_divergence

cudnn.benchmark = True

parser = argparse.ArgumentParser(description='Wide Residual Networks')
# Model options
parser.add_argument('--model', default='resnet', type=str)
parser.add_argument('--depth', default=28, type=int)
parser.add_argument('--width', default=10, type=float)
parser.add_argument('--dataset', default='CIFAR100', type=str)
parser.add_argument('--dataroot', type=str)
parser.add_argument('--dtype', default='float', type=str)
parser.add_argument('--groups', default=1, type=int)
parser.add_argument('--nthread', default=4, type=int)
parser.add_argument('--seed', default=1, type=int)

# Training options
parser.add_argument('--batch_size', default=128, type=int)
parser.add_argument('--lr', default=0.1, type=float)
parser.add_argument('--epochs', default=200, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--weight_decay', default=0.0005, type=float)
parser.add_argument('--epoch_step', default='[60,120,160]', type=str,
                    help='json list with epochs to drop lr on')
parser.add_argument('--lr_decay_ratio', default=0.2, type=float)
parser.add_argument('--resume', default='', type=str)
parser.add_argument('--note', default='', type=str)

# Device options
parser.add_argument('--cuda', action='store_true')
parser.add_argument('--save', default='', type=str,
                    help='save parameters and logs in this folder')
parser.add_argument('--ngpu', default=1, type=int,
                    help='number of GPUs to use for training')
parser.add_argument('--gpu_id', default='0', type=str,
                    help='id(s) for CUDA_VISIBLE_DEVICES')


def create_dataset(opt, train):
    transform = T.Compose([
        T.ToTensor(),
        T.Normalize(np.array([125.3, 123.0, 113.9]) / 255.0,
                    np.array([63.0, 62.1, 66.7]) / 255.0),
    ])
    if train:
        transform = T.Compose([
            T.Pad(4, padding_mode='reflect'),
            T.RandomHorizontalFlip(),
            T.RandomCrop(32),
            transform
        ])
    return getattr(datasets, opt.dataset)(opt.dataroot, train=train, download=True, transform=transform)

_outputs = []
_loss = []

def main():
    opt = parser.parse_args()
    print('parsed options:', vars(opt))
    epoch_step = json.loads(opt.epoch_step)
    num_classes = 10 if opt.dataset == 'CIFAR10' else 100

    torch.manual_seed(opt.seed)
    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu_id

    def create_iterator(mode):
        return DataLoader(create_dataset(opt, mode), opt.batch_size, shuffle=mode,
                          num_workers=opt.nthread, pin_memory=torch.cuda.is_available())

    train_loader = create_iterator(True)
    test_loader = create_iterator(False)

    f_1, params_1 = resnet(opt.depth, opt.width, num_classes)
    f_2, params_2 = resnet(opt.depth, opt.width, num_classes)

    def create_optimizer(opt, lr):
        print('creating optimizer with lr = ', lr)
        return SGD([v for v in params_1.values() if v.requires_grad] + [v for v in params_2.values() if v.requires_grad], lr, momentum=0.9, weight_decay=opt.weight_decay)

    optimizer = create_optimizer(opt, opt.lr)

    epoch = 0
    if opt.resume != '':
        raise NotImplementedError

    print('\nParameters:')
    print_tensor_dict(params_1)
    print_tensor_dict(params_2)

    n_parameters = sum([p.numel() for p in params_1.values() if p.requires_grad] + [p.numel() for p in params_2.values() if p.requires_grad])
    print('\nTotal number of parameters:', n_parameters)

    meter_loss = tnt.meter.AverageValueMeter()
    classacc = tnt.meter.ClassErrorMeter(accuracy=True)
    timer_train = tnt.meter.TimeMeter('s')
    timer_test = tnt.meter.TimeMeter('s')
    classacc_ep1 = tnt.meter.ClassErrorMeter(accuracy=True)
    classacc_ep2 = tnt.meter.ClassErrorMeter(accuracy=True)

    if not os.path.exists(opt.save):
        os.mkdir(opt.save)

    def h(sample):
        global _outputs, _loss

        connection_map = np.array([
            [0,0,0, 1,1,1],
            [0,0,0, 1,1,1],
            [0,0,0, 1,1,1],

            [1,1,1, 0,0,0],
            [1,1,1, 0,0,0],
            [1,1,1, 0,0,0]])

        inputs = cast(sample[0], opt.dtype)
        targets = cast(sample[1], 'long')
        net1_outputs = data_parallel(f_1, inputs, params_1, sample[2], list(range(opt.ngpu)))
        net2_outputs = data_parallel(f_2, inputs, params_2, sample[2], list(range(opt.ngpu)))
        net1_outputs = [o.float() for o in net1_outputs]
        net2_outputs = [o.float() for o in net2_outputs]

        _loss = []

        # hard supervision
        for i, o in enumerate(net1_outputs):
            _loss.append(F.cross_entropy(o, targets))

        for i, o in enumerate(net2_outputs):
            _loss.append(F.cross_entropy(o, targets))

        outputs = net1_outputs + net2_outputs
        # soft supervision
        for i, o in enumerate(outputs):
            for j, o2 in enumerate(outputs):
                if connection_map[i,j] > 0:
                    _loss.append(KL_divergence(o2.detach(),o))

        loss = sum(_loss)
        _outputs = net1_outputs

        return loss, net1_outputs[-1]

    def log(t, state):
        torch.save(dict(params=params_1, epoch=t['epoch'], optimizer=state['optimizer'].state_dict()),
                   os.path.join(opt.save, 'model.pt7'))
        z = {**vars(opt), **t}
        with open(os.path.join(opt.save, 'log.txt'), 'a') as flog:
            flog.write('json_stats: ' + json.dumps(z) + '\n')
        print(z)

    def on_sample(state):
        state['sample'].append(state['train'])

    def on_forward(state):
        loss = float(state['loss'])
        classacc.add(state['output'].data, state['sample'][1])
        classacc_ep1.add(_outputs[0].data, state['sample'][1])
        classacc_ep2.add(_outputs[1].data, state['sample'][1])
        meter_loss.add(loss)
        if state['train']:
            state['iterator'].set_postfix(loss=loss)

    def on_start(state):
        state['epoch'] = epoch

    def on_start_epoch(state):
        classacc.reset()
        classacc_ep1.reset()
        classacc_ep2.reset()
        meter_loss.reset()
        timer_train.reset()
        state['iterator'] = tqdm(train_loader, dynamic_ncols=True)

        epoch = state['epoch'] + 1
        if epoch in epoch_step:
            lr = state['optimizer'].param_groups[0]['lr']
            state['optimizer'] = create_optimizer(opt, lr * opt.lr_decay_ratio)

    def on_end_epoch(state):
        train_loss = meter_loss.value()
        train_acc = classacc.value()
        train_time = timer_train.value()
        train_acc_ep1 = classacc_ep1.value()
        train_acc_ep2 = classacc_ep2.value()

        meter_loss.reset()
        classacc.reset()
        timer_test.reset()
        classacc_ep1.reset()
        classacc_ep2.reset()

        with torch.no_grad():
            engine.test(h, test_loader)

        test_acc = classacc.value()[0]
        test_acc_ep1 = classacc_ep1.value()[0]
        test_acc_ep2 = classacc_ep2.value()[0]
        print(log({
            "train_loss": train_loss[0],
            "train_acc": train_acc[0],
            "test_loss": meter_loss.value()[0],
            "test_acc": test_acc,
            "train_acc_ep1": train_acc_ep1[0],
            "train_acc_ep2": train_acc_ep2[0],
            "test_acc_ep1": test_acc_ep1,
            "test_acc_ep2": test_acc_ep2,

            "epoch": state['epoch'],
            "num_classes": num_classes,
            "n_parameters": n_parameters,
            "train_time": train_time,
            "test_time": timer_test.value(),
        }, state))
        print('==> id: %s (%d/%d), test_acc: \33[91m%.2f\033[0m' %
              (opt.save, state['epoch'], opt.epochs, test_acc))

    engine = Engine()
    engine.hooks['on_sample'] = on_sample
    engine.hooks['on_forward'] = on_forward
    engine.hooks['on_start_epoch'] = on_start_epoch
    engine.hooks['on_end_epoch'] = on_end_epoch
    engine.hooks['on_start'] = on_start
    engine.train(h, train_loader, opt.epochs, optimizer)


if __name__ == '__main__':
    main()
