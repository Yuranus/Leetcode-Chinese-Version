"""
This file contains all operations about training saliency map using PyTorch
Paper: <The limitations of Deep Learning in Adversarial Settings>

Created by Kunhong Yu
Date: 2020/09/04
"""
import torch as t
import torchvision as tv
import matplotlib.pyplot as plt
from torch.nn import functional as F
import numpy as np
from configs import Config
from models import LeNet5_def
import random
import copy
import time
import sys

opt = Config()

def train_saliency_map(**kwargs):
    opt.parse(kwargs)
    opt.print()

    device = t.device('cuda') if opt.use_gpu else t.device('cpu')
    #Step 0 Decide the structure of the model#
    #Step 1 Load the data set#
    if opt.data_flag == 'mnist':
        dataset = tv.datasets.MNIST(root = opt.data_path,
                                    download = False,
                                    train = True,
                                    transform = opt.transform)
        num_classes = 10

    else:
        raise Exception('No other data sets!')

    #Step 2 Reshape the inputs#
    #Step 3 Normalize the inputs#
    dataloader = t.utils.data.DataLoader(dataset,
                                         batch_size = opt.batch_size,
                                         shuffle = True)

    #Step 4 Initialize parameters#
    #Step 5 Forward propagation(Vectorization/Activation functions)#
    if not opt.use_pretrained:
        model = LeNet5_def(num_classes = num_classes)
        model.to(device)

        #Step 6 Compute cost#
        cost = t.nn.CrossEntropyLoss().to(device)
        #Step 7 Backward propagation(Vectorization/Activation functions gradients)#
        if opt.optimizer == 'sgd':
            optimizer = t.optim.SGD(filter(lambda x : x.requires_grad, model.parameters()),
                                    lr = opt.init_lr,
                                    weight_decay = opt.weight_decay,
                                    momentum = 0.9,
                                    nesterov = True)

        elif opt.optimizer == 'adam':
            optimizer = t.optim.Adam(filter(lambda x : x.requires_grad, model.parameters()),
                                     lr = opt.init_lr,
                                     weight_decay = opt.weight_decay,
                                     amsgrad = True)

        else:
            raise Exception('No other optimizers!')

        #Step 8 Update parameters#
        lr_scheduler = t.optim.lr_scheduler.MultiStepLR(optimizer = optimizer,
                                                        milestones = opt.lr_decay_epochs,
                                                        gamma = opt.lr_decay_rate)

        costs = []
        accs = []
        for epoch in range(opt.epochs):
            print('Epoch : %d / %d.' % (epoch + 1, opt.epochs))
            print('Current epoch learning rate :', optimizer.param_groups[0]['lr'])
            for i, (batch_x, batch_y) in enumerate(dataloader):
                optimizer.zero_grad()
                batch_x = batch_x.to(device)
                batch_x = batch_x.view(batch_x.size(0), opt.image_channel, opt.image_height, opt.image_width)
                batch_y = batch_y.to(device)

                output = model(batch_x)
                batch_cost = cost(output, batch_y)
                batch_cost.backward()

                optimizer.step()

                if i % opt.batch_size == 0:
                    costs.append(batch_cost.item())
                    preds = t.max(output, dim = -1, keepdim = False)[1]
                    batch_acc = t.sum(preds == batch_y) / (1. * batch_x.size(0))
                    accs.append(batch_acc.item())
                    print('\tCurrent batch %d has cost : %.3f || Acc : %.2f%%.' % (i + 1, batch_cost.item(), batch_acc.item() * 100.))

            lr_scheduler.step()

        print('Training is done!')

        t.save(model, './models/sm/' + opt.data_flag + '_saved_model_' + opt.model_flag + '.pth')

        f, ax = plt.subplots(1, 2)
        f.suptitle('Loss & Acc')
        ax[0].plot(range(len(costs)), costs, label = 'cost')
        ax[0].set_title('Loss')
        ax[0].grid(True)
        ax[0].legend(loc = 'best')

        ax[1].plot(range(len(accs)), accs, label = 'acc')
        ax[1].set_title('Acc')
        ax[1].grid(True)
        ax[1].legend(loc = 'best')

        plt.savefig('./results/sm/' + opt.data_flag + '_saved_model_' + opt.model_flag + '.png')
        plt.close()

    else:
        print('We will generate adversarial examples using Salicency Map method using pretrained model!')
        if opt.data_flag == 'mnist':
            dataset = tv.datasets.MNIST(root = opt.data_path,
                                        download = False,
                                        train = False,
                                        transform = opt.transform)
            num_classes = 10

        else:
            raise Exception('No other datasets!')

        dataloader = t.utils.data.DataLoader(dataset,
                                             shuffle = False,
                                             batch_size = opt.batch_size)

        #Load pretrained model
        model = t.load('./models/sm/' + opt.data_flag + '_saved_model_' + opt.model_flag + '.pth')
        model.to(device)
        generate_num = 0

        if opt.num_adv_examples != 0:
            for i, (batch_x, batch_y) in enumerate(dataloader):
                sys.stdout.write('\r>>Generating %d / %d SM adversarial example.' % (generate_num + 1, opt.num_adv_examples))
                sys.stdout.flush()
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                batch_x = batch_x.view(batch_x.size(0), opt.image_channel, opt.image_height, opt.image_width)

                #we randomly select only one image from each batch
                index = random.choice(range(batch_x.size(0)))
                x_ori = batch_x[index : index + 1, ...]
                y_ori = batch_y[index : index + 1]
                #then, we predict using original image
                o_ori = model(x_ori)
                preds_ori = t.max(o_ori, dim = -1)
                prob_ori = F.softmax(o_ori, dim = -1)
                prob_ori = prob_ori[0, preds_ori[1].item()]

                #finally, we generate saliency map adversarial example
                #we randomly choose one target that is not equal to y_ori to be our attack target
                target = y_ori[0]
                while target == y_ori[0]:
                    target = random.choice(range(num_classes))
                    if target != y_ori[0]:
                        break

                s = time.time()
                x_adv = generate_saliency_map_adv_example(model, x_ori, target)
                e = time.time()
                sys.stdout.write(' || Generation elapsed time: ' + str(e - s) + 's.')
                sys.stdout.flush()
                #we predict using adversarial image
                o_adv = model(x_adv)
                preds_adv = t.max(o_adv, dim = -1)
                prob_adv = F.softmax(o_adv, dim = -1)
                prob_adv = prob_adv[0, preds_adv[1].item()]

                _, ax = plt.subplots(1, 2)
                x_ori = x_ori[0][0].data.cpu().numpy()
                x_adv = x_adv[0][0].data.cpu().numpy()
                ax[0].imshow((x_ori * 255.).astype('uint8'), cmap = 'gray')
                ax[0].axis('off')
                ax[0].grid(False)
                ax[0].set_title('Original\npred: ' + str(preds_ori[1][0].item()) + '||prob: ' + str(round(prob_ori.item(), 2) * 100.) + '%')

                ax[1].imshow((x_adv * 255.).astype('uint8'), cmap = 'gray')
                ax[1].axis('off')
                ax[1].grid(False)
                ax[1].set_title('Adv/Target: ' + str(target) + '\npred: ' + str(preds_adv[1][0].item()) + '||prob: ' + str(round(prob_adv.item(), 2) * 100.) + '%.')

                plt.savefig('./results/sm/adversarial_example_' + str(generate_num + 1) + '.png')
                generate_num += 1
                plt.close()

                if generate_num >= opt.num_adv_examples:
                    break

            print('\nDone!\n')

        else:
            print('No adversarial examples to be generated!')


def generate_saliency_map_adv_example(model, x, target, theta = 0.3, num_classes = 10):
    """This function is used to generate saliency map adversarial example
    This function is corresponding to Algorithm 1 in the original paper,
    we specify F_star to be pre-softmax layer
    Args :
        --model: Model instance
        --x: input original image
        --target: target class you want to predict
        --num_classes: how many classes
    return :
        --x_star: generated adv example
    """
    x_star = copy.deepcopy(x)
    x_star.requires_grad = True
    y_star_ = model(x_star)
    y_star = t.max(y_star_, dim = -1)[1].item()
    S = t.zeros_like(x_star, device = 'cpu')

    num_iter = 0
    max_iter = 100
    other_grad = 0.
    while y_star != target and num_iter < max_iter:
        for c in range(num_classes):
            if c == target:
                y_star_[:, target].backward(retain_graph = True)
                x_grad = x_star.grad.data.cpu().numpy().copy()
                x_star.grad.data.zero_()
                grad_t = t.Tensor(x_grad)
            else:
                y_star_[:, c].backward(retain_graph = True)
                x_grad = x_star.grad.data.cpu().numpy().copy()
                x_star.grad.data.zero_()
                other_grad += t.Tensor(x_grad)

        num_iter += 1

        S = t.where(grad_t < 0., t.full_like(S, 0.).to('cpu'), grad_t * abs(other_grad))
        S = t.where(other_grad > 0., t.full_like(S, 0.), S)
        S = S.view(-1)
        index = t.argmax(S)
        x_star.requires_grad = False
        x_star_ = x_star.view(-1)
        x_star_[index] += theta
        x_star = x_star_.view(x_star.size())

        x_star.requires_grad = True
        y_star_ = model(x_star)
        y_star = t.max(y_star_, dim = -1)[1].item()
        S = t.zeros_like(x_star)

    sys.stdout.write(' || Used iteration : ' + str(num_iter))
    sys.stdout.flush()

    return x_star