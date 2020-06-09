"""
This file contains buildings for Inception V1 (GoogLeNet) model using PyTorch without SIDE BRANCH in the original paper
Code inspired from : https://github.com/weiaicunzai/pytorch-cifar100/blob/master/models/googlenet.py

Created by Kunhong Yu
Date: 2020/06/08
"""
import torch as t
from models_def.basic_module import Basic_Module, Fire_Module
from utils import get_parameter_number
import time
from torch.nn import functional as F

class Inception_Module(t.nn.Module):
    def __init__(self, input_channels, n1x1, n3x3_reduce, n3x3, n5x5_reduce, n5x5, pool_proj):
        super().__init__()

        #1x1conv branch
        self.b1 = t.nn.Sequential(
            t.nn.Conv2d(input_channels, n1x1, kernel_size = 1),
            t.nn.BatchNorm2d(n1x1),
            t.nn.ReLU(inplace = True)
        )

        #1x1conv -> 3x3conv branch
        self.b2 = t.nn.Sequential(
            t.nn.Conv2d(input_channels, n3x3_reduce, kernel_size = 1),
            t.nn.BatchNorm2d(n3x3_reduce),
            t.nn.ReLU(inplace = True),
            t.nn.Conv2d(n3x3_reduce, n3x3, kernel_size = 3, padding = 1),
            t.nn.BatchNorm2d(n3x3),
            t.nn.ReLU(inplace = True)
        )

        #1x1conv -> 5x5conv branch
        #we use 2 3x3 conv filters stacked instead
        #of 1 5x5 filters to obtain the same receptive
        #field with fewer parameters
        self.b3 = t.nn.Sequential(
            t.nn.Conv2d(input_channels, n5x5_reduce, kernel_size = 1),
            t.nn.BatchNorm2d(n5x5_reduce),
            t.nn.ReLU(inplace = True),
            t.nn.Conv2d(n5x5_reduce, n5x5, kernel_size = 3, padding = 1),
            t.nn.BatchNorm2d(n5x5, n5x5),
            t.nn.ReLU(inplace = True),
            t.nn.Conv2d(n5x5, n5x5, kernel_size = 3, padding = 1),
            t.nn.BatchNorm2d(n5x5),
            t.nn.ReLU(inplace = True)
        )

        #3x3pooling -> 1x1conv
        #same conv
        self.b4 = t.nn.Sequential(
            t.nn.MaxPool2d(3, stride = 1, padding = 1),
            t.nn.Conv2d(input_channels, pool_proj, kernel_size = 1),
            t.nn.BatchNorm2d(pool_proj),
            t.nn.ReLU(inplace = True)
        )

    def forward(self, x):
        return t.cat([self.b1(x), self.b2(x), self.b3(x), self.b4(x)], dim = 1)

class GoogleNet(Basic_Module):

    def __init__(self,
                 num_classes = 100,
                 size = 4,
                 sigmoid_size = 3,
                 model_config = 'all',
                 data_flag = 'cifar100'):
        super(GoogleNet, self).__init__(size = size,
                                        sigmoid_size = sigmoid_size,
                                        teacher_channels = [512, 512, 528, 832, 832],
                                        student_channels = [256, 256, 384, 512, 512])

        if model_config.count('simple'):
            for parameter in super(GoogleNet, self).parameters():
                parameter.requires_grad = False

        ############################
        #         Teacher          #
        ############################
        self.prelayer = t.nn.Sequential(
            t.nn.Conv2d(3, 192, kernel_size = 3, padding = 1),
            t.nn.BatchNorm2d(192),
            t.nn.ReLU(inplace = True)
        )
        pre_params = get_parameter_number(self.prelayer)
        googlenet_params = get_parameter_number(self.prelayer)

        #although we only use 1 conv layer as prelayer,
        #we still use name a3, b3.......
        self.a3 = Inception_Module(192, 64, 96, 128, 16, 32, 32)#256
        a3_params = get_parameter_number(self.a3)
        googlenet_params['Trainable'] += a3_params['Trainable']
        googlenet_params['Total'] += a3_params['Total']

        self.b3 = Inception_Module(256, 128, 128, 192, 32, 96, 64)#480
        b3_params = get_parameter_number(self.b3)
        googlenet_params['Trainable'] += b3_params['Trainable']
        googlenet_params['Total'] += b3_params['Total']

        # """In general, an Inception network is a network consisting of
        # modules of the above type stacked upon each other, with occasional
        # max-pooling layers with stride 2 to halve the resolution of the
        # grid"""
        self.maxpool = t.nn.MaxPool2d(3, stride = 2, padding = 1)

        self.a4 = Inception_Module(480, 192, 96, 208, 16, 48, 64)#512
        a4_params = get_parameter_number(self.a4)
        googlenet_params['Trainable'] += a4_params['Trainable']
        googlenet_params['Total'] += a4_params['Total']

        self.b4 = Inception_Module(512, 160, 112, 224, 24, 64, 64)#512
        b4_params = get_parameter_number(self.b4)
        googlenet_params['Trainable'] += b4_params['Trainable']
        googlenet_params['Total'] += b4_params['Total']

        self.c4 = Inception_Module(512, 128, 128, 256, 24, 64, 64)#512
        c4_params = get_parameter_number(self.c4)
        googlenet_params['Trainable'] += c4_params['Trainable']
        googlenet_params['Total'] += c4_params['Total']

        self.d4 = Inception_Module(512, 112, 144, 288, 32, 64, 64)#528
        d4_params = get_parameter_number(self.d4)
        googlenet_params['Trainable'] += d4_params['Trainable']
        googlenet_params['Total'] += d4_params['Total']

        self.e4 = Inception_Module(528, 256, 160, 320, 32, 128, 128)#832
        e4_params = get_parameter_number(self.e4)
        googlenet_params['Trainable'] += e4_params['Trainable']
        googlenet_params['Total'] += e4_params['Total']

        self.a5 = Inception_Module(832, 256, 160, 320, 32, 128, 128)#832
        a5_params = get_parameter_number(self.a5)
        googlenet_params['Trainable'] += a5_params['Trainable']
        googlenet_params['Total'] += a5_params['Total']

        self.b5 = Inception_Module(832, 384, 192, 384, 48, 128, 128)#1024
        b5_params = get_parameter_number(self.b5)
        googlenet_params['Trainable'] += b5_params['Trainable']
        googlenet_params['Total'] += b5_params['Total']

        #input feature size: 8*8*1024
        self.avgpool = t.nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = t.nn.Dropout2d(p = 0.4)
        self.linear = t.nn.Linear(1024, num_classes)
        linear_params = get_parameter_number(self.linear)
        googlenet_params['Trainable'] += linear_params['Trainable']
        googlenet_params['Total'] += linear_params['Total']

        print('GoogLeNet total : ', googlenet_params)
        string = data_flag + '---' + model_config + ':\n'
        string += 'Teacher params : ' + str(googlenet_params) + '\n'

        ############################
        #         Student          #
        ############################
        self.additional_classifiers = []
        self.additional_classifiers.append(t.nn.Linear(256, num_classes))
        self.additional_classifiers.append(t.nn.Linear(256, num_classes))
        self.additional_classifiers.append(t.nn.Linear(384, num_classes))
        self.additional_classifiers.append(t.nn.Linear(512, num_classes))
        self.additional_classifiers.append(t.nn.Linear(512, num_classes))
        self.additional_classifiers = t.nn.ModuleList(self.additional_classifiers)

        self.student_model_conv1 = Fire_Module(in_channels = 3, squeeze_ratio = 0.125, ratio_3x3 = 0.5, expand_filters = 128)
        student_params = get_parameter_number(self.student_model_conv1)

        self.student_model_conv2 = Fire_Module(in_channels = 128, squeeze_ratio = 0.125, ratio_3x3 = 0.5, expand_filters = 256)
        conv2_params = get_parameter_number(self.student_model_conv2)
        student_params['Total'] += conv2_params['Total']
        student_params['Trainable'] += conv2_params['Trainable']
        student_params['Total'] += pre_params['Total']
        student_params['Trainable'] += pre_params['Trainable']
        student_params['Total'] += a3_params['Total']
        student_params['Trainable'] += a3_params['Trainable']
        student_params['Total'] += b3_params['Total']
        student_params['Trainable'] += b3_params['Trainable']
        student_params['Total'] += a4_params['Total']
        student_params['Trainable'] += a4_params['Trainable']
        student_params['Total'] += b4_params['Total']
        student_params['Trainable'] += b4_params['Trainable']
        se_params = get_parameter_number(self.ses_t[0])
        student_params['Total'] += se_params['Total']
        student_params['Trainable'] += se_params['Trainable']
        addi_params = get_parameter_number(self.additional_classifiers[0])
        student_params['Total'] += addi_params['Total']
        student_params['Trainable'] += addi_params['Trainable']
        print('Classifier 1 has params : ', student_params)
        string += 'Classifier 1 params : ' + str(student_params) + '\n'

        self.student_model_conv3 = Fire_Module(in_channels = 256, squeeze_ratio = 0.125, ratio_3x3 = 0.5, expand_filters = 256)
        conv3_params = get_parameter_number(self.student_model_conv3)
        student_params['Total'] -= addi_params['Total']
        student_params['Trainable'] -= addi_params['Trainable']
        student_params['Total'] += conv3_params['Total']
        student_params['Trainable'] += conv3_params['Trainable']
        student_params['Total'] += c4_params['Total']
        student_params['Trainable'] += c4_params['Trainable']
        se_params = get_parameter_number(self.ses_t[1])
        student_params['Total'] += se_params['Total']
        student_params['Trainable'] += se_params['Trainable']
        addi_params = get_parameter_number(self.additional_classifiers[1])
        student_params['Total'] += addi_params['Total']
        student_params['Trainable'] += addi_params['Trainable']
        print('Classifier 2 has params : ', student_params)
        string += 'Classifier 2 params : ' + str(student_params) + '\n'

        self.student_model_conv4 = Fire_Module(in_channels = 256, squeeze_ratio = 0.125, ratio_3x3 = 0.5, expand_filters = 384)
        conv4_params = get_parameter_number(self.student_model_conv4)
        student_params['Total'] -= addi_params['Total']
        student_params['Trainable'] -= addi_params['Trainable']
        student_params['Total'] += conv4_params['Total']
        student_params['Trainable'] += conv4_params['Trainable']
        student_params['Total'] += d4_params['Total']
        student_params['Trainable'] += d4_params['Trainable']
        se_params = get_parameter_number(self.ses_t[2])
        student_params['Total'] += se_params['Total']
        student_params['Trainable'] += se_params['Trainable']
        addi_params = get_parameter_number(self.additional_classifiers[2])
        student_params['Total'] += addi_params['Total']
        student_params['Trainable'] += addi_params['Trainable']
        print('Classifier 3 has params : ', student_params)
        string += 'Classifier 3 params : ' + str(student_params) + '\n'

        self.student_model_conv5 = Fire_Module(in_channels = 384, squeeze_ratio = 0.125, ratio_3x3 = 0.5, expand_filters = 512)
        conv5_params = get_parameter_number(self.student_model_conv5)
        student_params['Total'] -= addi_params['Total']
        student_params['Trainable'] -= addi_params['Trainable']
        student_params['Total'] += conv5_params['Total']
        student_params['Trainable'] += conv5_params['Trainable']
        student_params['Total'] += e4_params['Total']
        student_params['Trainable'] += e4_params['Trainable']
        se_params = get_parameter_number(self.ses_t[3])
        student_params['Total'] += se_params['Total']
        student_params['Trainable'] += se_params['Trainable']
        addi_params = get_parameter_number(self.additional_classifiers[3])
        student_params['Total'] += addi_params['Total']
        student_params['Trainable'] += addi_params['Trainable']
        print('Classifier 4 has params : ', student_params)
        string += 'Classifier 4 params : ' + str(student_params) + '\n'

        self.student_model_conv6 = Fire_Module(in_channels = 512, squeeze_ratio = 0.125, ratio_3x3 = 0.5, expand_filters = 512)
        conv6_params = get_parameter_number(self.student_model_conv6)
        student_params['Total'] -= addi_params['Total']
        student_params['Trainable'] -= addi_params['Trainable']
        student_params['Total'] += conv6_params['Total']
        student_params['Trainable'] += conv6_params['Trainable']
        student_params['Total'] += a5_params['Total']
        student_params['Trainable'] += a5_params['Trainable']
        se_params = get_parameter_number(self.ses_t[4])
        student_params['Total'] += se_params['Total']
        student_params['Trainable'] += se_params['Trainable']
        addi_params = get_parameter_number(self.additional_classifiers[4])
        student_params['Total'] += addi_params['Total']
        student_params['Trainable'] += addi_params['Trainable']
        print('Classifier 5 has params : ', student_params)
        string += 'Classifier 5 params : ' + str(student_params) + '\n'

        with open('./models_def/googlenet_params.txt', 'a+') as f:#save paramemters
            f.write(string)
            f.flush()

        self.model_config = model_config

    def forward(self, x, compute_se = False):
        students_out = []
        times = []

        #1. one step of teacher
        s = time.time()
        x_large1 = self.prelayer(x)
        x_large2 = self.a3(x_large1)
        x_large3 = self.b3(x_large2)
        x_large3 = self.maxpool(x_large3)
        x_large4 = self.a4(x_large3)
        x_large5 = self.b4(x_large4)
        #one step of student
        x_small1 = self.student_model_conv1(x)
        x_small2 = self.student_model_conv2(x_small1)
        if compute_se:
            t_sigmoid1 = self.get_outputs(x_large5, index = 0)
            x_small2 = x_small2 * t_sigmoid1

        x_small2 = F.max_pool2d(x_small2, 2)
        #We now need to specify student out to do classification#
        x_small2_ = F.adaptive_avg_pool2d(x_small2, (1, 1))
        x_small2_ = x_small2_.view(x_small2_.size(0), -1)
        students_out.append(self.additional_classifiers[0](x_small2_))
        e1 = time.time()
        dur1 = e1 - s
        times.append(dur1)

        #2. second step of teacher
        x_large6 = self.c4(x_large5)
        #second step of student
        x_small3 = self.student_model_conv3(x_small2)
        if compute_se:
            t_sigmoid2 = self.get_outputs(x_large6, index = 1)
            x_small3 = x_small3 * t_sigmoid2

        x_small3 = F.max_pool2d(x_small3, 2)
        #We now need to specify student out to do classification#
        x_small3_ = F.adaptive_avg_pool2d(x_small3, (1, 1))
        x_small3_ = x_small3_.view(x_small3_.size(0), -1)
        students_out.append(self.additional_classifiers[1](x_small3_))
        e2 = time.time()
        dur2 = e2 - s
        times.append(dur2)

        #3. third step of teacher
        x_large7 = self.d4(x_large6)
        #third step of student
        x_small4 = self.student_model_conv4(x_small3)
        if compute_se:
            t_sigmoid3 = self.get_outputs(x_large7, index = 2)
            x_small4 = x_small4 * t_sigmoid3

        x_small4 = F.max_pool2d(x_small4, 2)
        # We now need to specify student out to do classification#
        x_small4_ = F.adaptive_avg_pool2d(x_small4, (1, 1))
        x_small4_ = x_small4_.view(x_small4_.size(0), -1)
        students_out.append(self.additional_classifiers[2](x_small4_))
        e3 = time.time()
        dur3 = e3 - s
        times.append(dur3)

        #4. fourth step of teacher
        x_large8 = self.e4(x_large7)
        #fourth step of student
        x_small5 = self.student_model_conv5(x_small4)
        if compute_se:
            t_sigmoid4 = self.get_outputs(x_large8, index = 3)
            x_small5 = x_small5 * t_sigmoid4

        x_small5 = F.max_pool2d(x_small5, 2)
        # We now need to specify student out to do classification#
        x_small5_ = F.adaptive_avg_pool2d(x_small5, (1, 1))
        x_small5_ = x_small5_.view(x_small5_.size(0), -1)
        students_out.append(self.additional_classifiers[3](x_small5_))
        e4 = time.time()
        dur4 = e4 - s
        times.append(dur4)

        x_large8 = self.maxpool(x_large8)

        #5. fifth step of teacher
        x_large9 = self.a5(x_large8)
        #fifth step of student
        x_small6 = self.student_model_conv6(x_small5)
        if compute_se:
            t_sigmoid5 = self.get_outputs(x_large9, index = 4)
            x_small6 = x_small6 * t_sigmoid5

        x_small6 = F.max_pool2d(x_small6, 2)
        # We now need to specify student out to do classification#
        x_small6_ = F.adaptive_avg_pool2d(x_small6, (1, 1))
        x_small6_ = x_small6_.view(x_small6_.size(0), -1)
        students_out.append(self.additional_classifiers[4](x_small6_))
        e5 = time.time()
        dur5 = e5 - s
        times.append(dur5)

        #6. final steps of teacher
        x_large10 = self.b5(x_large9)
        x_large10 = self.avgpool(x_large10)
        x_large10 = self.dropout(x_large10)
        x_large10 = x_large10.view(x_large10.size()[0], -1)
        teacher_out = self.linear(x_large10)
        e6 = time.time()
        dur6 = e6 - s
        times.append(dur6)

        return teacher_out, students_out, times