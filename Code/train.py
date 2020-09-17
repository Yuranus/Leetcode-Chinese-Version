"""
This file is used to train the CRNN model using PyTorch
We add distillation procedure

Created by Kunhong Yu
Date: 2020/09/02
"""
import torch as t
import tqdm
from configs import Config
from utils import GetDataLoader
from model import CRNN_def, Distilled_CRNN_def
import matplotlib.pyplot as plt
from utils import model_info, cal_batch_acc
from utils import get_batch_label, train_test_split
from torch.nn import functional as F

opt = Config()

def train(**kwargs):
    """train the crnn model"""
    opt.parse(kwargs)
    opt.print_args()

    train_test_split(path = opt.data_path,
                     img_format = opt.img_format,
                     label_format = opt.label_format,
                     generating_again = opt.generating_again,
                     split_rate = opt.split_rate)

    device = t.device('cuda') if opt.use_gpu else t.device('cpu')
    #Step 0 Decide the structure of the model#
    #Step 1 Load the data set#
    dataset, dataloader = \
        GetDataLoader(path = opt.data_path,
                      train = True,
                      img_format = opt.img_format,
                      label_format = opt.label_format,
                      img_height = opt.img_height,
                      img_width = opt.img_width,
                      img_channels = opt.img_channels,
                      batch_size = opt.batch_size)
    #Step 2 Reshape the inputs#
    #Step 3 Normalize the inputs#
    #Step 4 Initialize parameters#
    #Step 5 Forward propagation(Vectorization/Activation functions)#
    crnn_model = CRNN_def(in_c = opt.img_channels,
                          feature_size = 512,
                          lstm_hidden = opt.lstm_hidden,
                          output_size = opt.output_size,
                          multilines = opt.multilines,
                          multisteps = opt.multisteps,
                          num_rows = opt.num_rows)
    crnn_model.to(device)
    distilled_crnn_model = Distilled_CRNN_def(in_c = opt.img_channels,
                                              feature_size = 512,
                                              lstm_hidden = opt.lstm_hidden,
                                              output_size = opt.output_size,
                                              multilines = opt.multilines,
                                              multisteps = opt.multisteps,
                                              num_rows = opt.num_rows)
    distilled_crnn_model.to(device)

    print('CRNN model : ')
    for name, parameters in crnn_model.named_parameters():
        print('\t', name, '...', parameters.requires_grad)

    print('Distilled CRNN model : ')
    for name, parameters in distilled_crnn_model.named_parameters():
        print('\t', name, '...', parameters.requires_grad)

    #Step 6 Compute cost#
    ctc_loss = t.nn.CTCLoss().to(device)#use CTC to derive the whole loss function
    #Step 7 Backward propagation(Vectorization/Activation functions gradients)#
    if opt.optimizer == 'sgd' or opt.optimizer == 'momentum' or opt.optimizer == 'nesterov':
        crnn_optimizer = t.optim.SGD(filter(lambda p: p.requires_grad, crnn_model.parameters()),
                                     lr = opt.init_lr,
                                     momentum = 0.9 if opt.optimizer == 'momentum' or opt.optimizer == 'nesterov' else 0.,
                                     nesterov = True if opt.optimizer == 'nesterov' else False,
                                     weight_decay = opt.weight_decay)
        distilled_crnn_optimizer = t.optim.SGD(filter(lambda p: p.requires_grad, distilled_crnn_model.parameters()),
                                               lr = opt.init_lr,
                                               momentum = 0.9 if opt.optimizer == 'momentum' or opt.optimizer == 'nesterov' else 0.,
                                               nesterov = True if opt.optimizer == 'nesterov' else False,
                                               weight_decay = opt.weight_decay)
    elif opt.optimizer == 'adam' or opt.optimizer == 'amsgrad':
        crnn_optimizer = t.optim.Adam(filter(lambda p: p.requires_grad, crnn_model.parameters()),
                                      lr = opt.init_lr,
                                      amsgrad = True if opt.optimizer == 'amsgrad' else False,
                                      weight_decay = opt.weight_decay)
        distilled_crnn_optimizer = t.optim.Adam(filter(lambda p: p.requires_grad, distilled_crnn_model.parameters()),
                                                lr = opt.init_lr,
                                                amsgrad = True if opt.optimizer == 'amsgrad' else False,
                                                weight_decay = opt.weight_decay)

    else:
        raise Exception('No other optimizers!')

    crnn_lr_schedule = t.optim.lr_scheduler.MultiStepLR(crnn_optimizer,
                                                        milestones = opt.lr_decay_epochs,
                                                        gamma = opt.lr_decay_rate)
    distilled_lr_schedule = t.optim.lr_scheduler.MultiStepLR(distilled_crnn_optimizer,
                                                             milestones = opt.lr_decay_epochs,
                                                             gamma = opt.lr_decay_rate)

    _ = model_info(crnn_model)
    _ = model_info(distilled_crnn_model)

    train_crnn_loss = []
    train_crnn_acc = []
    best_crnn_acc = 0.5#must have better accuracy than random guess of 0.5
    train_distilled_crnn_loss = []
    train_distilled_crnn_acc = []
    best_distilled_crnn_acc = 0.5#must have better accuracy than random guess of 0.5

    cd_loss = []
    lstm_loss = []
    h_loss = []
    c_loss = []
    softloss = []
    #Step 8 Update parameters#
    for epoch in tqdm.tqdm(range(opt.epochs)):
        print('Epoch : %d / %d.' % (epoch + 1, opt.epochs))
        print('Current epoch learning rate for CRNN: ', crnn_optimizer.param_groups[0]['lr'])
        if opt.distilled:
            print('Current epoch learning rate for Distilled_CRNN: ', distilled_crnn_optimizer.param_groups[0]['lr'])
        epoch_crnn_acc = 0.
        epoch_distilled_crnn_acc = 0.
        count = 0
        for i, (batch_x, index, path) in enumerate(dataloader):
            batch_x = batch_x.to(device)
            index = index.to(device)
            batch_x = batch_x.view(batch_x.size(0), opt.img_channels, opt.img_height, opt.img_width)

            crnn_optimizer.zero_grad()
            if not opt.multisteps:
                labels = get_batch_label(dataset, index)
                text, length = opt.converter.encode(labels)
                outputt, teachers, (hts, cts) = crnn_model(batch_x)
                #output has shape : [m, t, output_size]
                preds_size = [outputt.size(0)] * outputt.size(1)#batch_size * time_steps
                batch_crnn_cost = ctc_loss(outputt, text.to(t.long).to(device),
                                           t.IntTensor(preds_size).to(t.long).to(device),
                                           length.to(t.long).to(device))#ctc loss
            else:
                outputts, teachers, (htss, ctss) = crnn_model(batch_x)
                preds_size = [outputts[0].size(0)] * outputts[0].size(1)#batch_size * time_steps
                batch_crnn_cost = 0.
                labels = get_batch_label(dataset, index, multisteps = opt.multisteps, num_rows = opt.num_rows)
                for step in range(len(outputts)):
                    outputt = outputts[step]
                    label = labels[step]
                    text, length = opt.converter.encode(label)
                    batch_crnn_cost += ctc_loss(outputt, text.to(t.long).to(device),
                                                t.IntTensor(preds_size).to(t.long).to(device),
                                                length.to(t.long).to(device))#ctc loss

                batch_crnn_cost /= len(outputts)

            batch_crnn_cost.backward()
            crnn_optimizer.step()

            if opt.distilled:
                distilled_crnn_optimizer.zero_grad()
                if not opt.multisteps:
                    outputs, students, (hss, css) = distilled_crnn_model(batch_x)
                    #output has shape : [m, t, output_size]
                    preds_size = [outputs.size(0)] * outputs.size(1)#batch_size * time_steps
                else:
                    outputss, students, (hsss, csss) = distilled_crnn_model(batch_x)
                    preds_size = [outputss[0].size(0)] * outputss[0].size(1)#batch_size * time_steps

                #1. CTC loss
                if not opt.multisteps:
                    batch_distilled_crnn_cost = ctc_loss(outputs, text.to(t.long).to(device),
                                                         t.IntTensor(preds_size).to(t.long).to(device), length.to(t.long).to(device))

                else:
                    batch_ctc_loss = 0.
                    for step in range(len(outputss)):
                        outputs = outputss[step]
                        label = labels[step]
                        text, length = opt.converter.encode(label)
                        batch_ctc_loss += ctc_loss(outputs, text.to(t.long).to(device),
                                                   t.IntTensor(preds_size).to(t.long).to(device),
                                                   length.to(t.long).to(device))
                    batch_distilled_crnn_cost = batch_ctc_loss / (len(outputss) * 1.)

                #2. cd loss
                count_ = 0
                batch_cd_loss = 0.
                for teacher, student in zip(teachers, students):
                    batch_cd_loss += t.mean(t.pow(teacher - student, 2)).to(device)
                    count_ += 1
                batch_cd_loss /= count_

                batch_distilled_crnn_cost += opt.alpha * batch_cd_loss

                #3. lstm loss
                #3.1 H values
                count_ = 0
                cur_lossh = 0.
                if not opt.multisteps:
                    for ht, hs in zip(hts, hss):
                        cur_lossh += t.mean(t.pow(ht - hs, 2)).to(device)
                        count_ += 1
                else:
                    for hts, hss in zip(htss, hsss):
                        cur_loss = 0.
                        q = 0.
                        for ht, hs in zip(hts, hss):
                            cur_loss += t.mean(t.pow(ht - hs, 2)).to(device)
                            q += 1.

                        cur_lossh += cur_loss / q
                        count_ += 1
                cur_lossh /= count_
                #3.2 C values
                cur_lossc = 0.
                count_ = 0
                if not opt.multisteps:
                    for ct, cs in zip(cts, css):
                        cur_lossc += t.mean(t.pow(ct - cs, 2)).to(device)
                        count_ += 1
                else:
                    for cts, css in zip(ctss, csss):
                        cur_loss = 0.
                        q = 0.
                        for ct, cs in zip(cts, css):
                            cur_loss += t.mean(t.pow(ct - cs, 2)).to(device)
                            q += 1.

                        cur_lossc += cur_loss / q
                        count_ += 1
                cur_lossc /= count_
                batch_lstm_loss = (cur_lossc + cur_lossh) / 2.
                batch_distilled_crnn_cost += opt.beta * batch_lstm_loss

                #4. soft loss
                if not opt.multisteps:
                    batch_softloss = -t.mean(t.sum(F.softmax(outputt.detach() / opt.temperature, dim = 1) * \
                                                    t.log(F.softmax(outputs / opt.temperature, dim = 1) + 1e-10),
                                                    dim = 1)).to(device)
                else:
                    batch_softloss = 0.
                    for outputt, outputs in zip(outputts, outputss):
                        batch_softloss += -t.mean(t.sum(F.softmax(outputt.detach() / opt.temperature, dim = 1) * \
                                                        t.log(F.softmax(outputs / opt.temperature, dim = 1) + 1e-10),
                                                        dim = 1)).to(device)
                    batch_softloss /= len(outputts)

                batch_distilled_crnn_cost += opt.gamma * batch_softloss

                batch_distilled_crnn_cost.backward()
                distilled_crnn_optimizer.step()

            if i % opt.batch_size == 0:
                count += 1
                train_crnn_loss.append(batch_crnn_cost.item())
                crnn_model.eval()
                batch_crnn_acc, predictions = cal_batch_acc(crnn_model, opt.converter, batch_x, labels, level = opt.level)

                print('\nCRNN samples predictions: ')
                print('=' * 30)
                print('Labels : ', label)
                print('*' * 20)
                print('Predictions : ', predictions)
                print('=' * 30)
                crnn_model.train()
                train_crnn_acc.append(batch_crnn_acc)

                if opt.distilled:
                    train_distilled_crnn_loss.append(batch_distilled_crnn_cost.item())
                    cd_loss.append(opt.alpha * batch_cd_loss.item())
                    lstm_loss.append(opt.beta * batch_lstm_loss.item())
                    h_loss.append(opt.beta * cur_lossh.item())
                    c_loss.append(opt.beta * cur_lossc.item())
                    softloss.append(opt.gamma * batch_softloss.item())
                    distilled_crnn_model.eval()
                    batch_distilled_crnn_acc, predictions = cal_batch_acc(distilled_crnn_model, opt.converter, batch_x,
                                                                          label, level = opt.level)

                    print('=' * 50)
                    print('Distilled CRNN samples predictions : ')
                    print('=' * 30)
                    print('Labels : ', label)
                    print('*' * 20)
                    print('Predictions : ', predictions)
                    print('=' * 30)

                    distilled_crnn_model.train()
                    train_distilled_crnn_acc.append(batch_distilled_crnn_acc)

                print('\tCRNN : ')
                print('\tBatch %d has crnn cost : %.3f.|| Accuracy : ' % (i + 1, batch_crnn_cost.item()), end = '')
                if isinstance(batch_crnn_acc, tuple):
                    print('Character-level acc : %.2f%%; Image-level acc : %.2f%%.' % (batch_crnn_acc[0] * 100., batch_crnn_acc[1] * 100.))
                    combined_acc = (2. * batch_crnn_acc[0] * batch_crnn_acc[1]) / (batch_crnn_acc[0] + batch_crnn_acc[1] + 1e-7)#f1
                    epoch_crnn_acc += combined_acc
                else:
                    if opt.level == 'char':
                        print('Character-level acc : %.2f%%.' % (batch_crnn_acc * 100.))
                    elif opt.level == 'whole':
                        print('Image-level acc : %.2f%%.' % (batch_crnn_acc * 100.))
                    else:
                        raise Exception('No other levels!')

                    epoch_crnn_acc += batch_crnn_acc

                if opt.distilled:
                    print('\tDistilled : ')
                    print('\tBatch %d has distilled crnn cost : %.3f.[softloss %3f & cd loss %.3f & lstm loss %.3f & h_loss %.3f & c_loss %.3f]. --> \n\t\tAccuracy : '
                            % (i + 1, batch_distilled_crnn_cost.item(), opt.gamma * batch_softloss.item(),
                               opt.alpha * batch_cd_loss.item(), opt.beta * batch_lstm_loss.item(),
                               opt.beta * cur_lossh.item(), opt.beta * cur_lossc.item()), end = '')
                    if isinstance(batch_distilled_crnn_acc, tuple):
                        print('Character-level acc : %.2f%%; Image-level acc : %.2f%%.' % (
                        batch_distilled_crnn_acc[0] * 100., batch_distilled_crnn_acc[1] * 100.))
                        combined_acc = (2. * batch_distilled_crnn_acc[0] * batch_distilled_crnn_acc[1]) / (
                                    batch_distilled_crnn_acc[0] + batch_distilled_crnn_acc[1] + 1e-7)  # f1
                        epoch_distilled_crnn_acc += combined_acc
                    else:
                        if opt.level == 'char':
                            print('Character-level acc : %.2f%%.' % (batch_distilled_crnn_acc * 100.))
                        elif opt.level == 'whole':
                            print('Image-level acc : %.2f%%.' % (batch_distilled_crnn_acc * 100.))
                        else:
                            raise Exception('No other levels!')

                        epoch_distilled_crnn_acc += batch_distilled_crnn_acc

        epoch_crnn_acc /= count
        epoch_distilled_crnn_acc /= count

        print('This epoch has crnn acc : {:.2f}%.'.format(epoch_crnn_acc * 100.))
        if opt.save_best_model:
            if epoch % opt.save_best_model_iter == 0:
                if epoch_crnn_acc > best_crnn_acc:
                    best_crnn_acc = epoch_crnn_acc
                    t.save(crnn_model, './checkpoints/save_best_train_crnn_model_epoch_%d_%s.pkl' % (epoch + 1, opt.model_config))
                else:
                    print('This epoch has no improvement on training accuracy on crnn model, skipping saving the model!')

        if opt.distilled:
            print('This epoch has distilled crnn acc : {:.2f}%.'.format(epoch_distilled_crnn_acc * 100.))
            if opt.save_best_model:
                if epoch % opt.save_best_model_iter == 0:
                    if epoch_distilled_crnn_acc > best_distilled_crnn_acc:
                        best_distilled_crnn_acc = epoch_distilled_crnn_acc
                        t.save(distilled_crnn_model, './checkpoints/save_best_train_distilled_crnn_model_epoch_%d_%s.pkl' % (
                                    epoch + 1, opt.model_config))
                    else:
                        print('This epoch has no improvement on training accuracy on distilled crnn model, skipping saving the model!')

        crnn_lr_schedule.step()
        distilled_lr_schedule.step()

    t.save(crnn_model, './checkpoints/final_crnn_model_%s.pkl' % opt.model_config)

    f, ax = plt.subplots(1, 2)
    f.suptitle('Useful statistics for CRNN')
    ax[0].plot(range(len(train_crnn_loss)), train_crnn_loss, label = 'CRNN training loss')
    ax[0].grid(True)
    ax[0].set_title('CRNN training loss')
    ax[0].legend(loc = 'best')

    if isinstance(train_crnn_acc[0], tuple):
        char_acc = [c_acc[0] for c_acc in train_crnn_acc]
        whole_acc = [c_acc[1] for c_acc in train_crnn_acc]
        ax[1].plot(range(len(char_acc)), char_acc, label = 'Character-level acc')
        ax[1].plot(range(len(whole_acc)), whole_acc, label = 'Image-level acc')

    else:
        if opt.level == 'char':
            ax[1].plot(range(len(train_crnn_acc)), train_crnn_acc, label = 'Character-level acc')
        elif opt.level == 'whole':
            ax[1].plot(range(len(train_crnn_acc)), train_crnn_acc, label = 'Image-level acc')
        else:
            raise Exception('No other levels!')

    ax[1].grid(True)
    ax[1].set_title('CRNN training acc')
    ax[1].legend(loc = 'best')

    plt.savefig('./results/training_crnn_statistics_%s.png' % opt.model_config)
    plt.close()

    if opt.distilled:
        t.save(distilled_crnn_model, './checkpoints/final_distilled_crnn_model_%s.pkl' % opt.model_config)

        f, ax = plt.subplots(1, 5)
        f.suptitle('Useful statistics for Distilled CRNN')
        ax[0].plot(range(len(train_distilled_crnn_loss)), train_distilled_crnn_loss, label = 'Distilled CRNN training loss')
        ax[0].grid(True)
        ax[0].set_title('Distilled CRNN training loss')
        ax[0].legend(loc = 'best')

        if isinstance(train_distilled_crnn_acc[0], tuple):
            char_acc = [c_acc[0] for c_acc in train_distilled_crnn_acc]
            whole_acc = [c_acc[1] for c_acc in train_distilled_crnn_acc]
            ax[1].plot(range(len(char_acc)), char_acc, label= ' Character-level acc')
            ax[1].plot(range(len(whole_acc)), whole_acc, label = 'Image-level acc')

        else:
            if opt.level == 'char':
                ax[1].plot(range(len(train_distilled_crnn_acc)), train_distilled_crnn_acc, label = 'Character-level acc')
            elif opt.level == 'whole':
                ax[1].plot(range(len(train_distilled_crnn_acc)), train_distilled_crnn_acc, label = 'Image-level acc')
            else:
                raise Exception('No other levels!')

        ax[1].grid(True)
        ax[1].set_title('Distilled training acc')
        ax[1].legend(loc = 'best')

        ax[2].plot(range(len(cd_loss)), cd_loss, label = 'Distilled CRNN training cd loss')
        ax[2].grid(True)
        ax[2].set_title('Distilled CRNN training cd loss')
        ax[2].legend(loc = 'best')

        ax[3].plot(range(len(softloss)), softloss, label = 'Distilled CRNN training soft loss')
        ax[3].grid(True)
        ax[3].set_title('Distilled CRNN training soft loss')
        ax[3].legend(loc = 'best')

        ax[4].plot(range(len(lstm_loss)), lstm_loss, label = 'Distilled CRNN training lstm loss')
        ax[4].plot(range(len(h_loss)), h_loss, label = 'Distilled CRNN training lstm hidden loss')
        ax[4].plot(range(len(c_loss)), c_loss, label = 'Distilled CRNN training lstm cell loss')
        ax[4].grid(True)
        ax[4].set_title('Distilled CRNN training lstm loss')
        ax[4].legend(loc = 'best')

        plt.savefig('./results/training_distilled_crnn_statistics_%s.png' % opt.model_config)
        plt.close()

    print('Training is done!\n')