import torch
import torch.nn as nn
from model.densenet import DenseNet
from utils.options import args
import torch.optim as optim
import utils.common as utils

import os
import time
import copy
import sys
import random
import numpy as np
import heapq
from tqdm import tqdm
from data import cifar10,cifar100
from importlib import import_module

checkpoint = utils.checkpoint(args)
device = torch.device(f"cuda:{args.gpus[0]}") if torch.cuda.is_available() else 'cpu'
logger = utils.get_logger(os.path.join(args.job_dir + 'logger.log'))
loss_func = nn.CrossEntropyLoss()

conv_num_cfg = {
    'vgg16': 13,
    'resnet56' : 27,
    'resnet110' : 54,
    'googlenet' : 9,
    'densenet':36,
}

pso_dimension = conv_num_cfg[args.cfg]

# load Data
print('==> Loading Data..')
if args.data_set == 'cifar10':
    loader = cifar10.Data(args)
elif args.data_set == 'cifar100':
    loader = cifar100.Data(args)
# else:
#     loader = imagenet.Data(args)

# flops and params calculators
Calculator = utils.flops_params_calculator(data_set=args.data_set,device=device)


# Model
print('==> Loading Model..')
if args.arch == 'vgg_cifar':
    origin_model = import_module(f'model.{args.arch}').VGG(args.cfg).to(device)
elif args.arch == 'resnet_cifar':
    origin_model = import_module(f'model.{args.arch}').resnet(args.cfg).to(device)
elif args.arch == 'googlenet':
    origin_model = import_module(f'model.{args.arch}').googlenet().to(device)
elif args.arch == 'densenet':
    origin_model = import_module(f'model.{args.arch}').densenet().to(device)

if args.pso_model is None or not os.path.exists(args.pso_model):
    raise ('pso_model path should be exist!')


ckpt = torch.load(args.pso_model, map_location=device)
origin_model.load_state_dict(ckpt['state_dict'])
oristate_dict = origin_model.state_dict()

# Define partical

class Partical():
    def __init__(self):
        super(Partical,self).__init__()
        self.code = [] # conv num of each layer
        self.fitness = 0
        self.velocity = []

best_partical = Partical()
best_partical_state = {}


# load pretrain model


def load_vgg_partical_model(model, random_rule):
    #print(ckpt['state_dict'])
    global oristate_dict
    state_dict = model.state_dict()
    last_select_index = None #Conv index selected in the previous layer

    for name, module in model.named_modules():

        if isinstance(module, nn.Conv2d):

            oriweight = oristate_dict[name + '.weight']
            curweight = state_dict[name + '.weight']
            orifilter_num = oriweight.size(0)
            currentfilter_num = curweight.size(0)

            if orifilter_num != currentfilter_num and (random_rule == 'random_pretrain' or random_rule == 'l1_pretrain'):

                select_num = currentfilter_num
                if random_rule == 'random_pretrain':
                    select_index = random.sample(range(0, orifilter_num-1), select_num)
                    select_index.sort()
                else:
                    l1_sum = list(torch.sum(torch.abs(oriweight), [1, 2, 3]))
                    select_index = list(map(l1_sum.index, heapq.nlargest(currentfilter_num, l1_sum)))
                    select_index.sort()
                if last_select_index is not None:
                    for index_i, i in enumerate(select_index):
                        for index_j, j in enumerate(last_select_index):
                            state_dict[name + '.weight'][index_i][index_j] = \
                                oristate_dict[name + '.weight'][i][j]
                else:
                    for index_i, i in enumerate(select_index):
                        state_dict[name + '.weight'][index_i] = \
                            oristate_dict[name + '.weight'][i]

                last_select_index = select_index

            else:
                state_dict[name + '.weight'] = oriweight
                last_select_index = None

    model.load_state_dict(state_dict)

def load_google_partical_model(model, random_rule):
    global oristate_dict
    state_dict = model.state_dict()
        
    last_select_index = None
    all_honey_conv_name = []
    all_honey_bn_name = []

    for name, module in model.named_modules():

        if isinstance(module, nn.Inception):

            honey_filter_channel_index = ['.branch5x5.3']  # the index of sketch filter and channel weight
            honey_channel_index = ['.branch3x3.3', '.branch5x5.6']  # the index of sketch channel weight
            honey_filter_index = ['.branch3x3.0', '.branch5x5.0']  # the index of sketch filter weight
            honey_bn_index = ['.branch3x3.1', '.branch5x5.1', '.branch5x5.4'] #the index of sketch bn weight
            
            for bn_index in honey_bn_index:
                all_honey_bn_name.append(name + bn_index)

            for weight_index in honey_filter_channel_index:
                last_select_index = None
                conv_name = name + weight_index + '.weight'
                all_honey_conv_name.append(name + weight_index)

                oriweight = oristate_dict[conv_name]
                curweight = state_dict[conv_name]

                orifilter_num = oriweight.size(1)
                currentfilter_num = curweight.size(1)

                if orifilter_num != currentfilter_num and (random_rule == 'random_pretrain' or random_rule == 'l1_pretrain'):

                    select_num = currentfilter_num
                    if random_rule == 'random_pretrain':
                        select_index = random.sample(range(0, orifilter_num-1), select_num)
                        select_index.sort()
                    else:
                        l1_sum = list(torch.sum(torch.abs(oriweight), [1, 2, 3]))
                        select_index = list(map(l1_sum.index, heapq.nlargest(currentfilter_num, l1_sum)))
                        select_index.sort()

                    #print(state_dict[conv_name].size())
                    #print(oristate_dict[conv_name].size())

                


                orifilter_num = oriweight.size(0)
                currentfilter_num = curweight.size(0)


                select_index_1 = copy.deepcopy(select_index)


                if orifilter_num != currentfilter_num and (random_rule == 'random_pretrain' or random_rule == 'l1_pretrain'):

                    select_num = currentfilter_num
                    if random_rule == 'random_pretrain':
                        select_index = random.sample(range(0, orifilter_num-1), select_num)
                        select_index.sort()
                    else:
                        l1_sum = list(torch.sum(torch.abs(oriweight), [1, 2, 3]))
                        select_index = list(map(l1_sum.index, heapq.nlargest(currentfilter_num, l1_sum)))
                        select_index.sort()

                for index_i, i in enumerate(select_index):
                    for index_j, j in enumerate(select_index_1):
                            state_dict[conv_name][index_i][index_j] = \
                                oristate_dict[conv_name][i][j]



            for weight_index in honey_channel_index:

                conv_name = name + weight_index + '.weight'
                all_honey_conv_name.append(name + weight_index)

                oriweight = oristate_dict[conv_name]
                curweight = state_dict[conv_name]
                orifilter_num = oriweight.size(1)
                currentfilter_num = curweight.size(1)

                #print(state_dict[conv_name].size())
                #print(oristate_dict[conv_name].size())


                if orifilter_num != currentfilter_num and (random_rule == 'random_pretrain' or random_rule == 'l1_pretrain'):

                    select_num = currentfilter_num
                    if random_rule == 'random_pretrain':
                        select_index = random.sample(range(0, orifilter_num-1), select_num)
                        select_index.sort()
                    else:
                        l1_sum = list(torch.sum(torch.abs(oriweight), [1, 2, 3]))
                        select_index = list(map(l1_sum.index, heapq.nlargest(currentfilter_num, l1_sum)))
                        select_index.sort()


                    for i in range(state_dict[conv_name].size(0)):
                        for index_j, j in enumerate(select_index):
                            state_dict[conv_name][i][index_j] = \
                                oristate_dict[conv_name][i][j]


            for weight_index in honey_filter_index:

                conv_name = name + weight_index + '.weight'
                all_honey_conv_name.append(name + weight_index)
                oriweight = oristate_dict[conv_name]
                curweight = state_dict[conv_name]

                orifilter_num = oriweight.size(0)
                currentfilter_num = curweight.size(0)

                if orifilter_num != currentfilter_num and (random_rule == 'random_pretrain' or random_rule == 'l1_pretrain'):

                    select_num = currentfilter_num
                    if random_rule == 'random_pretrain':
                        select_index = random.sample(range(0, orifilter_num-1), select_num)
                        select_index.sort()
                    else:
                        l1_sum = list(torch.sum(torch.abs(oriweight), [1, 2, 3]))
                        select_index = list(map(l1_sum.index, heapq.nlargest(currentfilter_num, l1_sum)))
                        select_index.sort()

                    for index_i, i in enumerate(select_index):
                            state_dict[conv_name][index_i] = \
                                oristate_dict[conv_name][i]


    for name, module in model.named_modules(): #Reassign non sketch weights to the new network

        if isinstance(module, nn.Conv2d):

            if name not in all_honey_conv_name:
                state_dict[name + '.weight'] = oristate_dict[name + '.weight']
                state_dict[name + '.bias'] = oristate_dict[name + '.bias']

        elif isinstance(module, nn.BatchNorm2d):

            if name not in all_honey_bn_name:
                state_dict[name + '.weight'] = oristate_dict[name + '.weight']
                state_dict[name + '.bias'] = oristate_dict[name + '.bias']
                state_dict[name + '.running_mean'] = oristate_dict[name + '.running_mean']
                state_dict[name + '.running_var'] = oristate_dict[name + '.running_var']

        elif isinstance(module, nn.Linear):
            state_dict[name + '.weight'] = oristate_dict[name + '.weight']
            state_dict[name + '.bias'] = oristate_dict[name + '.bias']

    model.load_state_dict(state_dict)

def load_dense_partical_model(model, random_rule):

    global oristate_dict
    

    state_dict = model.state_dict()

    conv_weight = []
    conv_trans_weight = []
    bn_weight = []
    bn_bias = []

    for i in range(3):
        for j in range(12):
            conv1_weight_name = 'dense%d.%d.conv1.weight' % (i + 1, j)
            conv_weight.append(conv1_weight_name)

            bn1_weight_name = 'dense%d.%d.bn1.weight' % (i + 1, j)
            bn_weight.append(bn1_weight_name)

            bn1_bias_name = 'dense%d.%d.bn1.bias' %(i+1,j)
            bn_bias.append(bn1_bias_name)

    for i in range(2):
        conv1_weight_name = 'trans%d.conv1.weight' % (i + 1)
        conv_weight.append(conv1_weight_name)
        conv_trans_weight.append(conv1_weight_name)

        bn_weight_name = 'trans%d.bn1.weight' % (i + 1)
        bn_weight.append(bn_weight_name)

        bn_bias_name = 'trans%d.bn1.bias' % (i + 1)
        bn_bias.append(bn_bias_name)
    
    bn_weight.append('bn.weight')
    bn_bias.append('bn.bias')


    
    for k in range(len(conv_weight)):
        conv_weight_name = conv_weight[k]
        oriweight = oristate_dict[conv_weight_name]
        curweight = state_dict[conv_weight_name]
        orifilter_num = oriweight.size(1)
        currentfilter_num = curweight.size(1)
        select_num = currentfilter_num
        #print(orifilter_num)
        #print(currentfilter_num)

        if orifilter_num != currentfilter_num and (random_rule == 'random_pretrain' or random_rule == 'l1_pretrain'):
            if random_rule == 'random_pretrain':
                select_index = random.sample(range(0, orifilter_num-1), select_num)
                select_index.sort()
            else:
                l1_sum = list(torch.sum(torch.abs(oriweight), [1, 2, 3]))
                select_index = list(map(l1_sum.index, heapq.nlargest(currentfilter_num, l1_sum)))
                select_index.sort()

            for i in range(curweight.size(0)):
                for index_j, j in enumerate(select_index):
                    state_dict[conv_weight_name][i][index_j] = \
                            oristate_dict[conv_weight_name][i][j]


    for k in range(len(bn_weight)):

        bn_weight_name = bn_weight[k]
        bn_bias_name = bn_bias[k]
        bn_bias.append(bn_bias_name)
        bn_weight.append(bn_weight_name)
        oriweight = oristate_dict[bn_weight_name]
        curweight = state_dict[bn_weight_name]

        orifilter_num = oriweight.size(0)
        currentfilter_num = curweight.size(0)
        select_num = currentfilter_num

        if orifilter_num != currentfilter_num and (random_rule == 'random_pretrain' or random_rule == 'l1_pretrain'):
            if random_rule == 'random_pretrain':
                select_index = random.sample(range(0, orifilter_num-1), select_num)
                select_index.sort()
            else:
                l1_sum = list(torch.sum(torch.abs(oriweight), [1, 2, 3]))
                select_index = list(map(l1_sum.index, heapq.nlargest(currentfilter_num, l1_sum)))
                select_index.sort()

            for index_j, j in enumerate(select_index):
                state_dict[bn_weight_name][index_j] = \
                        oristate_dict[bn_weight_name][j]
                state_dict[bn_bias_name][index_j] = \
                        oristate_dict[bn_bias_name][j]

    oriweight = oristate_dict['fc.weight']
    curweight = state_dict['fc.weight']
    orifilter_num = oriweight.size(1)
    currentfilter_num = curweight.size(1)
    select_num = currentfilter_num

    if orifilter_num != currentfilter_num and (random_rule == 'random_pretrain' or random_rule == 'l1_pretrain'):
        if random_rule == 'random_pretrain':
            select_index = random.sample(range(0, orifilter_num-1), select_num)
            select_index.sort()
        else:
            l1_sum = list(torch.sum(torch.abs(oriweight), [1, 2, 3]))
            select_index = list(map(l1_sum.index, heapq.nlargest(currentfilter_num, l1_sum)))
            select_index.sort()

        for i in range(curweight.size(0)): 
            for index_j, j in enumerate(select_index):
                state_dict['fc.weight'][i][index_j] = \
                        oristate_dict['fc.weight'][i][j]



    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            conv_name = name + '.weight'
            if conv_name not in conv_weight:
                state_dict[conv_name] = oristate_dict[conv_name]

        elif isinstance(module, nn.BatchNorm2d):
            bn_weight_name = name + '.weight'
            bn_bias_name = name + '.bias'
            if bn_weight_name not in bn_weight and bn_bias_name not in bn_bias:
                state_dict[bn_weight_name] = oristate_dict[bn_weight_name]
                state_dict[bn_bias_name] = oristate_dict[bn_bias_name]

    model.load_state_dict(state_dict)


def load_resnet_partical_model(model, random_rule):

    cfg = { 
           'resnet56': [9,9,9],
           'resnet110': [18,18,18],
           }

    global oristate_dict
    state_dict = model.state_dict()
        
    current_cfg = cfg[args.cfg]
    last_select_index = None

    all_honey_conv_weight = []

    for layer, num in enumerate(current_cfg):
        layer_name = 'layer' + str(layer + 1) + '.'
        for k in range(num):
            for l in range(2):
                conv_name = layer_name + str(k) + '.conv' + str(l+1)
                conv_weight_name = conv_name + '.weight'
                all_honey_conv_weight.append(conv_weight_name)
                oriweight = oristate_dict[conv_weight_name]
                curweight = state_dict[conv_weight_name]
                orifilter_num = oriweight.size(0)
                currentfilter_num = curweight.size(0)
                #logger.info('weight_num {}'.format(conv_weight_name))
                #logger.info('orifilter_num {}\tcurrentnum {}\n'.format(orifilter_num,currentfilter_num))
                #logger.info('orifilter  {}\tcurrent {}\n'.format(oristate_dict[conv_weight_name].size(),state_dict[conv_weight_name].size()))

                if orifilter_num != currentfilter_num and (random_rule == 'random_pretrain' or random_rule == 'l1_pretrain'):

                    select_num = currentfilter_num
                    if random_rule == 'random_pretrain':
                        select_index = random.sample(range(0, orifilter_num-1), select_num)
                        select_index.sort()
                    else:
                        l1_sum = list(torch.sum(torch.abs(oriweight), [1, 2, 3]))
                        select_index = list(map(l1_sum.index, heapq.nlargest(currentfilter_num, l1_sum)))
                        select_index.sort()
                    if last_select_index is not None:
                        logger.info('last_select_index'.format(last_select_index))
                        for index_i, i in enumerate(select_index):
                            for index_j, j in enumerate(last_select_index):
                                state_dict[conv_weight_name][index_i][index_j] = \
                                    oristate_dict[conv_weight_name][i][j]
                    else:
                        for index_i, i in enumerate(select_index):
                            state_dict[conv_weight_name][index_i] = \
                                oristate_dict[conv_weight_name][i]  

                    last_select_index = select_index
                    #logger.info('last_select_index{}'.format(last_select_index)) 

                elif last_select_index != None:
                    for index_i in range(orifilter_num):
                        for index_j, j in enumerate(last_select_index):
                            state_dict[conv_weight_name][index_i][index_j] = \
                                oristate_dict[conv_weight_name][index_i][j]
                    last_select_index = None

                else:
                    state_dict[conv_weight_name] = oriweight
                    last_select_index = None

    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            conv_name = name + '.weight'
            if conv_name not in all_honey_conv_weight:
                state_dict[conv_name] = oristate_dict[conv_name]

        elif isinstance(module, nn.Linear):
            state_dict[name + '.weight'] = oristate_dict[name + '.weight']
            state_dict[name + '.bias'] = oristate_dict[name + '.bias']

    #for param_tensor in state_dict:
        #logger.info('param_tensor {}\tType {}\n'.format(param_tensor,state_dict[param_tensor].size()))
    #for param_tensor in model.state_dict():
        #logger.info('param_tensor {}\tType {}\n'.format(param_tensor,model.state_dict()[param_tensor].size()))
 

    model.load_state_dict(state_dict)


# fine-tune
def train(model,optimizer, trainLoader, args, epoch):

    model.train()
    losses = utils.AverageMeter()
    accuracy = utils.AverageMeter() # calculate average

    print_freq = len(trainLoader.dataset) // args.train_batch_size // 10

    start_time = time.time()
    for batch,(inputs,targets) in enumerate(trainLoader):
        inputs,targets = (inputs).to(device), targets.to(device)
        optimizer.zero_grad()
        output = model(inputs)
        loss = loss_func(output, targets)
        loss.backward()
        losses.update(loss.item(), inputs.size(0))
        optimizer.step()

        prec1 = utils.accuracy(output, targets)
        accuracy.update(prec1[0], inputs.size(0))

        if batch % print_freq == 0 and batch != 0:
            current_time = time.time()
            cost_time = current_time - start_time
            logger.info(
                'Epoch[{}] ({}/{}):\t'
                'Loss {:.4f}\t'
                'Accurary {:.2f}%\t\t'
                'Time {:.2f}s'.format(
                    epoch, batch * args.train_batch_size, len(trainLoader.dataset),
                    float(losses.avg), float(accuracy.avg), cost_time
                )
            )
            start_time = current_time

#Testing
def test(model,testLoader):

    model.eval()

    losses = utils.AverageMeter()
    accurary = utils.AverageMeter()

    start_time = time.time()

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testLoader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = loss_func(outputs, targets)

            losses.update(loss.item(), inputs.size(0))
            predicted = utils.accuracy(outputs, targets)
            accurary.update(predicted[0], inputs.size(0))

        current_time = time.time()
        logger.info(
            'Test Loss {:.4f}\tAccurary {:.2f}%\t\tTime {:.2f}s\n'
            .format(float(losses.avg), float(accurary.avg), (current_time - start_time))
        )
    return accurary.avg


# fitness calculation (PASE)

## Adaptive BN(from https://github.com/whut2962575697/naic_reid/blob/a474f368d967f4c9a7a87eded2467f93f585a7e1/utils/swa.py#L11)
def _check_bn(module, flag):
    if issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm):
        flag[0] = True

def check_bn(model):
    flag = [False]
    model.apply(lambda module: _check_bn(module, flag))
    return flag[0]

def reset_bn(module):
    if issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm):
        # momentum = module.momentum # reset momentum
        module.reset_running_stats()
        # module.momentum = momentum

def _get_momenta(module, momenta):
    if issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm):
        momenta[module] = module.momentum

def _set_momenta(module, momenta):
    if issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm):
        module.momentum = momenta[module]

def bn_update(model,loader,cumulative = False):
    """
        BatchNorm buffers update (if any).
        Performs 1 epochs to estimate buffers average using train dataset.
        :param model: model being update
        :param loader: train dataset loader for buffers average estimation.
        :param cumulative: cumulative moving average or exponential moving average
        :return: approcimate train accuracy (util.AverageMeter)
    """
    if not check_bn(model):
        return

    print("approcimate process:")

    train_acc = utils.AverageMeter()
    model.train()
    model.apply(reset_bn)

    if cumulative:
        momenta = {}
        model.apply(lambda module: _get_momenta(module, momenta))
        for module in momenta.keys():
            module.momentum = None
    approcimate_num = int(len(loader)*args.approcimate_rate)
    with torch.no_grad(): # freeze all the parameters
        # with tqdm(total=approcimate_num) as pbar:
        for i,(inputs,targets) in enumerate(loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            prec1 = utils.accuracy(outputs,targets)
            # pbar.set_description(f"train accuracy:{prec1}")
            train_acc.update(prec1[0],inputs.size(0))
            # pbar.update(1)
            if i >= approcimate_num:
                break
    return train_acc



def calcuate_fitness(partical,train_loader,args):
    global best_partical
    global best_partical_state
    print(f'partical code [{partical}]')
    if args.arch == 'vgg_cifar':
        model = import_module(f'model.{args.arch}').BeeVGG(args.cfg,honeysource=partical).to(device)
        load_vgg_partical_model(model, args.random_rule)
    elif args.arch == 'resnet_cifar':
        model = import_module(f'model.{args.arch}').resnet(args.cfg,honey=partical).to(device)
        load_resnet_partical_model(model, args.random_rule)
    elif args.arch == 'googlenet':
        model = import_module(f'model.{args.arch}').googlenet(honey=partical).to(device)
        load_google_partical_model(model, args.random_rule)
    elif args.arch == 'densenet':
        model = import_module(f'model.{args.arch}').densenet(honey=partical).to(device)
        load_dense_partical_model(model, args.random_rule)

    start_time = time.time()
    if len(args.gpus) != 1:
        model = nn.DataParallel(model, device_ids=args.gpus)
    
    # optimizer = optim.SGD(model.parameters(), lr = arg.lr, momentum=args.momentum, weight_decay= args.weight_decay)
    #PASE
    
    if args.pase_enable == True:
        train_acc = bn_update(model,train_loader)
        train_acc = float(train_acc.avg)

    else:
    # Fine-tune
    #start_time = time.time()
        train_acc = utils.AverageMeter()
        if len(args.gpus) != 1:
            model = nn.DataParallel(model, device_ids=args.gpus)

        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

        #test(model, loader.testLoader)

        model.train()
        for epoch in range(args.calfitness_epoch):
            for batch, (inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(device), targets.to(device)
                #print("ok")
                optimizer.zero_grad()
                output = model(inputs)
                loss = loss_func(output, targets)
                loss.backward()
                optimizer.step()

                prec1 = utils.accuracy(output, targets)
                train_acc.update(prec1[0], inputs.size(0))
        train_acc = float(train_acc.avg)
    fit_acc = utils.AverageMeter()
    model.eval()
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(loader.testLoader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            predicted = utils.accuracy(outputs, targets)
            fit_acc.update(predicted[0], inputs.size(0))
    current_time = time.time()
    logger.info(
            'Partical Source fintness {:.2f}%\t\ttrain accuracy {:.2f}%\t\tTime {:.2f}s\n'
            .format(float(fit_acc.avg), train_acc,(current_time - start_time))
        )
    
    return fit_acc.avg


# PSO algorithm
C1 = args.individual_factor
C2 = args.global_factor

def get_partical_group(nums=10):

    partical_group=[]
    for i in range(nums):
        p = Partical()
        for _ in range(pso_dimension):
            p.code.append(copy.deepcopy(random.randint(args.min_preserve,args.max_preserve)))
            p.velocity.append(copy.deepcopy(random.randint(-int(args.max_vel),int(args.max_vel))))
        p.fitness = calcuate_fitness(p.code,loader.trainLoader,args)
        print(f'Partical No.{i+1}, fitness={p.fitness}')

        partical_group.append(p)
    
    return partical_group

best_group = []
pso_inertia = args.pso_inertia[0]
inertia_max = args.pso_inertia[0]
inertia_min = args.pso_inertia[1]

# partical_group = []
def get_init_best_pop(partical_group:list):
    # global best
    global best_partical
    global best_group
    best_partical = copy.deepcopy(max(partical_group,key=lambda x:x.fitness))
    best_group = copy.deepcopy(partical_group)
    # tmp_group = copy.deepcopy(partical_group)


def update_vel(partical_group:list):
    global best_partical
    global best_group

    for i in range(args.partical_number):
        param2change = np.random.randint(0,pso_dimension-1,args.partical_change_num)
        R1 = np.random.uniform(0,1,args.partical_change_num)
        R2 = np.random.uniform(0,1,args.partical_change_num)

        for j in range(args.partical_change_num):
            tmp = int(pso_inertia*partical_group[i].velocity[param2change[j]] + \
                (best_group[i].code[param2change[j]]-partical_group[i].code[param2change[j]])*R1[j]*C1 + \
                (best_partical.code[param2change[j]]-partical_group[i].code[param2change[j]])*R2[j]*C2
            )

            if tmp > args.max_vel:
                tmp = args.max_vel
            elif tmp < -args.max_vel:
                tmp = -args.max_vel
            partical_group[i].velocity[param2change[j]] = tmp

def update_code(partical_group:list):

    for i in range(args.partical_number):
        
        for j in range(pso_dimension):
            partical_group[i].code[j]+=partical_group[i].velocity[j]
            if partical_group[i].code[j] > args.max_preserve:
                partical_group[i].code[j] = args.max_preserve
            elif partical_group[i].code[j] < args.min_preserve:
                partical_group[i].code[j] = args.min_preserve

def update_fitness(partical_group:list):

    for p in partical_group:
        p.fitness = calcuate_fitness(p.code,loader.trainLoader,args)


def PSO_loop():
    global best_partical
    global pso_inertia,inertia_max,inertia_min
    start_time = time.time()
    pso_start_time = time.time()
    print('==>start PsoPruning...')
    
    # initialize
    print('==>Initialize PSO')
    partical_group = get_partical_group(args.partical_number)
    get_init_best_pop(partical_group)

    print('get particals ')

    print('==>Run PSO')
    for cycle in range(args.max_cycle):
        current_time = time.time()
        logger.info(
            'Search Cycle [{}]\t Best partical Source {}\tBest partical Source fitness {:.2f}%\tTime {:.2f}s\n'
            .format(cycle, best_partical.code, float(best_partical.fitness), (current_time - start_time))
        )
        start_time = time.time()

        update_vel(partical_group)

        update_code(partical_group)

        update_fitness(partical_group)

        for i in range(args.partical_number):
            if partical_group[i].fitness > best_group[i].fitness:
                best_group[i] = copy.deepcopy(partical_group[i])
        best_partical = copy.deepcopy(max(best_group,key=lambda x:x.fitness))

        # update inertia
        pso_inertia = inertia_max - (inertia_max-inertia_min)*((cycle/args.max_cycle)**2)
        
    pso_end_time = time.time()
    print('==> PSO pruning Complete!')

    logger.info(
        'Best partical Source {}\tBest partical Source fitness {:.2f}%\tTime Used{:.2f}s\n'
        .format(best_partical.code, float(best_partical.fitness), (pso_end_time - pso_start_time))
    )

def random_loop():
    global best_partical
    start_time = time.time()
    rnd_start_time = time.time()
    print('==>start Pruning by random prune radio...')
    
    # initialize
    print('==>Initialize partical')

    partical = Partical()

    random_code = np.random.randint(args.min_preserve,args.max_preserve,pso_dimension)
    partical.code = list(random_code)
    partical.fitness = calcuate_fitness(partical.code,loader.trainLoader,args)
    best_partical = copy.deepcopy(partical)

    F = open(args.job_dir+'code_record.txt','w')
    print('==>Run pruning')
    for cycle in range(args.max_cycle*args.partical_number): #equal loops to pso
        current_time = time.time()
        logger.info(
            'Search Cycle [{}]\t Best partical Source {}\tBest partical Source fitness {:.2f}%\tTime {:.2f}s\n'
            .format(cycle, best_partical.code, float(best_partical.fitness), (current_time - start_time))
        )
        start_time = time.time()
        random_code = np.random.randint(args.min_preserve,args.max_preserve,pso_dimension)
        partical.code = list(random_code)
        partical.fitness = calcuate_fitness(partical.code,loader.trainLoader,args)
        F.write(f'{partical.code},{partical.fitness}\n')
        if(partical.fitness > best_partical.fitness):
            best_partical = copy.deepcopy(partical)
    rnd_end_time = time.time()
    
    logger.info(
        'Best partical Source {}\tBest partical Source fitness {:.2f}%\tTime Used{:.2f}s\n'
        .format(best_partical.code, float(best_partical.fitness), (rnd_end_time - rnd_start_time))
    )
    F.close()

def main():
    global best_partical
    global pso_inertia,inertia_max,inertia_min

    start_epoch=0
    best_acc = 0.0
    code = []

    if args.resume == None:
        test(origin_model,loader.testLoader)

        if args.best_partical == None:
            # start_time = time.time()
            # pso_start_time = time.time()
            # print('==>start PsoPruning...')
            
            # # initialize
            # print('==>Initialize PSO')
            # partical_group = get_partical_group(args.partical_number)
            # get_init_best_pop(partical_group)

            # print('get particals ')

            # print('==>Run PSO')
            # for cycle in range(args.max_cycle):
            #     current_time = time.time()
            #     logger.info(
            #         'Search Cycle [{}]\t Best partical Source {}\tBest partical Source fitness {:.2f}%\tTime {:.2f}s\n'
            #         .format(cycle, best_partical.code, float(best_partical.fitness), (current_time - start_time))
            #     )
            #     start_time = time.time()

            #     update_vel(partical_group)

            #     update_code(partical_group)

            #     update_fitness(partical_group)

            #     for i in range(args.partical_number):
            #         if partical_group[i].fitness > best_group[i].fitness:
            #             best_group[i] = copy.deepcopy(partical_group[i])
            #     best_partical = copy.deepcopy(max(best_group,key=lambda x:x.fitness))

            #     # update inertia
            #     pso_inertia = inertia_max - (inertia_max-inertia_min)*((cycle/args.max_cycle)**2)
                
            # pso_end_time = time.time()
            # print('==> PSO pruning Complete!')

            # logger.info(
            #     'Best partical Source {}\tBest partical Source fitness {:.2f}%\tTime Used{:.2f}s\n'
            #     .format(best_partical.code, float(best_partical.fitness), (pso_end_time - pso_start_time))
            # )
            if args.random_enable == True:
                random_loop()
            else:
                PSO_loop()
        else:
            best_partical.code = args.best_partical
        code = best_partical.code
    # prune the Model
        print('==> Building model..')
        if args.arch == 'vgg_cifar':
            model = import_module(f'model.{args.arch}').BeeVGG(args.cfg,honeysource=code).to(device)
            load_vgg_partical_model(model, args.random_rule)
        elif args.arch == 'resnet_cifar':
            model = import_module(f'model.{args.arch}').resnet(args.cfg,honey=code).to(device)
            load_resnet_partical_model(model, args.random_rule)
        elif args.arch == 'googlenet':
            model = import_module(f'model.{args.arch}').googlenet(honey=code).to(device)
            load_google_partical_model(model, args.random_rule)
        elif args.arch == 'densenet':
            model = import_module(f'model.{args.arch}').densenet(honey=code).to(device)
            load_dense_partical_model(model, args.random_rule)


        checkpoint.save_honey_model(model.state_dict())
        print(args.random_rule + ' Done!')

        if len(args.gpus) > 1:
            model = nn.DataParallel(model,device_ids=args.gpus)

        # if args.best_partical == None:
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.lr_decay_step, gamma=0.1)
    else:
        resumeckpt = torch.load(args.resume)
        state_dict = resumeckpt['state_dict']
        code = resumeckpt['partical_code']
        # code = [9, 9, 9, 5, 5, 8, 7, 9, 3, 6, 9, 5, 1]
    # prune the Model
        print('==> Building model..')
        if args.arch == 'vgg_cifar':
            model = import_module(f'model.{args.arch}').BeeVGG(args.cfg, honeysource=code).to(device)
        elif args.arch == 'resnet_cifar':
            model = import_module(f'model.{args.arch}').resnet(args.cfg,honey=code).to(device)
        elif args.arch == 'googlenet':
            model = import_module(f'model.{args.arch}').googlenet(honey=code).to(device)
        elif args.arch == 'densenet':
            model = import_module(f'model.{args.arch}').densenet(honey=code).to(device)

        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.lr_decay_step, gamma=0.1)

        model.load_state_dict(state_dict)
        optimizer.load_state_dict(resumeckpt['optimizer'])
        scheduler.load_state_dict(resumeckpt['scheduler'])
        start_epoch = resumeckpt['epoch']

        if len(args.gpus) != 1:
            model = nn.DataParallel(model, device_ids=args.gpus)

    if args.test_only:
        test(model, loader.testLoader)

    # fine-tune
    else: 
        for epoch in range(start_epoch, args.num_epochs):
            train(model, optimizer, loader.trainLoader, args, epoch)
            scheduler.step()
            test_acc = test(model, loader.testLoader)

            is_best = best_acc < test_acc
            best_acc = max(best_acc, test_acc)

            model_state_dict = model.module.state_dict() if len(args.gpus) > 1 else model.state_dict()

            state = {
                'state_dict': model_state_dict,
                'best_acc': best_acc,
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'epoch': epoch + 1,
                'partical_code': code
            }
            checkpoint.save_model(state, epoch + 1, is_best)

        logger.info('Best accurary: {:.3f}'.format(float(best_acc)))

if __name__ == '__main__':
    main()