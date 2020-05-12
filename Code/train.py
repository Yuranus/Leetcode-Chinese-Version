"""
This file contains all operations about training models using PyTorch

Created by Kunhong Yu
Date: 2020/04/23
"""
import torch as t
import torchvision as tv
from torch.nn import functional as F
from config import Config
from models import *
from utils import vis_images, vis_procedure
import tqdm
import os
from sklearn.externals import joblib

opt = Config()

def train_models(**kwargs):
    """This function is used to train the model"""
    opt.parse(kwargs)
    opt.print_configs()

    device = t.device('cuda') if opt.use_gpu else t.device('cpu')
    #Step 0 Decide the structure of the model#
    #Step 1 Load the data set#
    #Step 2 Reshape the inputs#
    #Step 3 Normalize the inputs#
    t.manual_seed(opt.training_seed)
    if opt.data_flag == 'mnist':
        train_dataset = tv.datasets.MNIST(root = opt.data_path,
                                          train = True,
                                          download = False,
                                          transform = opt.transforms_)
    elif opt.data_flag == 'cifar10':
        train_dataset = tv.datasets.CIFAR10(root = opt.data_path + 'CIFAR10',
                                            train = True,
                                            download = False,
                                            transform = opt.transforms_)

    elif opt.data_flag == 'cifar100':
        train_dataset = tv.datasets.CIFAR100(root = opt.data_path + 'CIFAR100',
                                             train = True,
                                             download = False,
                                             transform = opt.transforms_)

    elif opt.data_flag == 'imagenette320':
        train_dataset = tv.datasets.ImageFolder(root = opt.data_path + os.sep + 'imagenette/v320' + os.sep + 'train',
                                                transform = opt.transforms_)

    elif opt.data_flag == 'imagewoof320':
        train_dataset = tv.datasets.ImageFolder(root = opt.data_path + os.sep + 'imagewoof/v320' + os.sep + 'train',
                                                transform = opt.transforms_)

    elif opt.data_flag == 'tiny_in':
        train_dataset = tv.datasets.ImageFolder(root = opt.data_path + os.sep + 'tiny_imagenet' + os.sep + 'train',
                                                transform = opt.transforms_)
        #print(train_dataset[0][0].size())

        class2idx = train_dataset.class_to_idx
        joblib.dump(class2idx, opt.data_path + os.sep + 'tiny_imagenet' + os.sep + 'class2idx.pkl')

    else:
        raise Exception('No other data set!')

    vis_images(opt, train_dataset, num_images = 36)

    train_loader = t.utils.data.DataLoader(train_dataset,
                                           shuffle = True,
                                           batch_size = opt.batch_size)

    #Step 4 Initialize parameters#
    #Step 5 Forward propagation(Vectorization/Activation functions)#
    if opt.data_flag == 'mnist':#mnist
        model = MNISTModel_def(size = opt.size,
                               sigmoid_size = opt.sigmoid_size_ratio,
                               model_config = opt.model_config,
                               clipped_value = opt.clipped_value).to(device)

    elif opt.data_flag.startswith('cifar'):#cifar10 or cifar100
        if opt.model_str == 'resnet34':
            model = ResNet34_VGG16_def(size = opt.size,
                                       sigmoid_size = opt.sigmoid_size_ratio,
                                       num_classes = 10 if opt.data_flag == 'cifar10' else 100,
                                       model_config = opt.model_config,
                                       clipped_value = opt.clipped_value,
                                       resnet34_pretrained = opt.resnet34_pretrained,
                                       vgg16_pretrained = opt.vgg16_pretrained).to(device)
        else:
            model = CIFARModel_def(size = opt.size,
                                   sigmoid_size = opt.sigmoid_size_ratio,
                                   output_size = 10 if opt.data_flag == 'cifar10' else 100,
                                   model_config = opt.model_config,
                                   clipped_value = opt.clipped_value).to(device)

    elif opt.data_flag == 'imagenette320' or opt.data_flag == 'imagewoof320':#imagenette320/imagewoof320
        model = ResNet34_VGG16_def(size = opt.size,
                                   sigmoid_size = opt.sigmoid_size_ratio,
                                   num_classes = 10,
                                   model_config = opt.model_config,
                                   clipped_value = opt.clipped_value,
                                   resnet34_pretrained = opt.resnet34_pretrained,
                                   vgg16_pretrained = opt.vgg16_pretrained).to(device)

    elif opt.data_flag == 'tiny_in':
        model = ResNet34_VGG16_def(size = opt.size,
                                   sigmoid_size = opt.sigmoid_size_ratio,
                                   num_classes = 200,
                                   model_config = opt.model_config,
                                   clipped_value = opt.clipped_value,
                                   resnet34_pretrained = opt.resnet34_pretrained,
                                   vgg16_pretrained = opt.vgg16_pretrained).to(device)
    else:
        raise Exception('No other data set!')

    for name, parameters in model.named_parameters():
        print(name, '...', parameters.requires_grad)

    #Step 6 Compute cost#
    hard_loss = t.nn.CrossEntropyLoss().to(device)
    #soft_loss defined in the dynamic graph
    #Step 7 Backward propagation(Vectorization/Activation functions gradients)#
    optimizer = t.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                             lr = opt.init_lr, amsgrad = True, weight_decay = opt.weight_decay)

    lr_optimizer = t.optim.lr_scheduler.MultiStepLR(optimizer = optimizer,
                                                    milestones = opt.lr_change_epochs,
                                                    gamma = opt.learning_rate_decay)#learning rate decay

    #Step 8 Update parameters#
    teacher_hard_loss_ = []
    student_hard_loss_ = []
    soft_loss_ = []
    teacher_accs_ = []
    student_accs_ = []
    best_acc = 0
    for epoch in tqdm.tqdm(range(opt.epochs)):
        epoch_acc = 0
        count = 0
        print('Epoch %d / %d.' % (epoch + 1, opt.epochs))
        print('Learning rate : ', optimizer.param_groups[0]['lr'])
        for i, (batch_x, batch_y) in enumerate(train_loader):
            #print(batch_y)
            batch_soft_loss = 0.

            optimizer.zero_grad()
            batch_x = batch_x.to(device)
            batch_x = batch_x.view(batch_x.size(0), opt.image_channel, opt.image_height, opt.image_width)
            batch_y = batch_y.to(device)
            teacher_out, student_out = model(batch_x)

            #Compute whole cost#
            batch_teacher_hard_loss = hard_loss(teacher_out, batch_y)
            batch_student_hard_loss = hard_loss(student_out, batch_y)
            teacher_hard_loss_.append(batch_teacher_hard_loss.item())
            student_hard_loss_.append(batch_student_hard_loss.item())
            if opt.alpha != 0.:
                batch_soft_loss = -t.mean(t.sum(F.softmax(teacher_out / opt.temperature, dim = 1) * \
                                                t.log(F.softmax(student_out / opt.temperature, dim = 1) + 1e-10),
                                                dim = 1)).to(device)
                soft_loss_.append(opt.alpha * batch_soft_loss.item())

            #whole loss#
            batch_loss = (batch_teacher_hard_loss + batch_student_hard_loss) + opt.alpha * batch_soft_loss

            batch_loss.backward()
            optimizer.step()

            if i % opt.batch_size == 0:
                model.eval()
                teacher_batch_acc, student_batch_acc = cal_batch_acc(model, batch_x, batch_y)
                model.train()
                teacher_accs_.append(teacher_batch_acc)
                student_accs_.append(student_batch_acc)
                batch_acc = 2 * teacher_batch_acc * student_batch_acc / (teacher_batch_acc + student_batch_acc + 1e-7)
                epoch_acc += batch_acc
                count += 1
                if opt.model_config == 'simple' or opt.model_config == 'se':
                    print('\tBatch %d has loss : %.3f.--->\n\t\tteacher loss : %.3f || student loss : %.3f.--->teacher acc : %.2f%% || student acc : %.2f%%.' % \
                          (i + 1, batch_loss.item(), batch_teacher_hard_loss.item(), batch_student_hard_loss.item(),
                          teacher_batch_acc * 100, student_batch_acc * 100))

                elif opt.model_config == 'soft' or opt.model_config == 'all':
                    print('\tBatch %d has loss : %.3f.--->\n\t\tteacher loss : %.3f || student loss : %.3f || soft loss : %.3f.--->teacher acc : %.2f%% || student acc : %.2f%%.' % \
                          (i + 1, batch_loss.item(), batch_teacher_hard_loss.item(), batch_student_hard_loss.item(),
                          opt.alpha * batch_soft_loss.item(), teacher_batch_acc * 100, student_batch_acc * 100))
                else:
                    raise Exception('No other model!')

        lr_optimizer.step()
        epoch_acc = (epoch_acc * 100) / count
        print('This epoch has accuracy : {:.2f}%.'.format(epoch_acc))
        if epoch_acc > best_acc:
            best_acc = epoch_acc
            t.save(model, './models/' + opt.data_flag + '/training_saved_best_model_' + opt.model_config + '_' + str(epoch + 1) + '_' + opt.model_str + '.pkl')

    print('Training is done!')

    t.save(model, './models/' + opt.data_flag + '/saved_model_' + opt.model_config + '_' +opt.model_str + '.pkl')

    if opt.model_config == 'simple' or opt.model_config == 'se':
        vis_procedure(teacher_hard_loss = teacher_hard_loss_,
                      student_hard_loss = student_hard_loss_,
                      teacher_acc = teacher_accs_,
                      student_acc = student_accs_,
                      opt = opt)

    if opt.model_config == 'soft' or opt.model_config == 'all':
        vis_procedure(teacher_hard_loss = teacher_hard_loss_,
                      student_hard_loss = student_hard_loss_,
                      soft_loss = soft_loss_,
                      teacher_acc = teacher_accs_,
                      student_acc = student_accs_,
                      opt = opt)

def cal_batch_acc(model, data, labels):
    """This function is used to calculate batch accuracy
    Args :
        --model: model instance, already load onto GPU if opt.use_gpu == True
        --data: input data, already load onto GPU if opt.use_gpu == True
        --labels: ground-truth labels, already load onto GPU if opt.use_gpu == True
    return :
        --teacher's batch accuracy and studnet's batch accuracy
    """
    batch_teacher_out, batch_student_out = model(data)
    teacher_preds = t.max(batch_teacher_out, 1)[1].data
    student_preds = t.max(batch_student_out, 1)[1].data
    batch_acc_teacher = t.sum(teacher_preds == labels) / (data.size(0) * 1.)
    batch_acc_student = t.sum(student_preds == labels) / (data.size(0) * 1.)
    if opt.use_gpu:
        batch_acc_teacher = batch_acc_teacher.cpu()
        batch_acc_student = batch_acc_student.cpu()

    return batch_acc_teacher.item(), batch_acc_student.item()