"""
This file contains all operations about implementing utilities functions using PyTorch

Created by Kunhong Yu
Date: 2020/04/20
"""
import torch as t
import torchvision as tv
import matplotlib.pyplot as plt
import math
import numpy as np
from collections import OrderedDict
from shutil import move, copyfile, rmtree
import os
from sklearn.externals import joblib
import sys
from PIL import Image

def vis_images(opt, dataset, num_images):
    """This function is used to visualize images
    Args :
        --opt: Config instance
        --dataset
        --num_images: how many images you want to visualize
    """
    if opt.data_flag == 'mnist' or opt.data_flag.startswith('cifar'):
        if opt.data_flag == 'mnist':
            imgs = dataset.data[:num_images].numpy()
            imgs = imgs.reshape(imgs.shape[0], 1, 28, 28)
        elif opt.data_flag.startswith('cifar'):
            imgs = dataset.data[:num_images]
            imgs = imgs.transpose(0, 3, 1, 2)

        show_img = tv.utils.make_grid(t.from_numpy(imgs),
                                      nrow = int(math.sqrt(num_images)))
        plt.imshow(np.asarray(show_img).transpose(1, 2, 0).astype('uint8'))
        plt.axis('off')
        plt.show()
        plt.close()

    elif opt.data_flag.startswith('tiny_in'):
            pass

def vis_procedure(**kwargs):
    """This function is used to visualize training procedure with loss function and accuracy"""
    opt = kwargs['opt']
    results = []
    titles = ['teacher_hard_loss', 'student_hard_loss']
    if 'soft_loss' in kwargs.keys():
        titles.append('soft_loss')

    teacher_hard_loss = kwargs['teacher_hard_loss']
    results.append(teacher_hard_loss)
    student_hard_loss = kwargs['student_hard_loss']
    results.append(student_hard_loss)
    if opt.alpha != 0.:
        soft_loss = kwargs['soft_loss']
        results.append(soft_loss)

    teacher_accs = kwargs['teacher_acc']
    student_accs = kwargs['student_acc']

    ########################################
    #                 LOSS                 #
    ########################################
    if len(results) == 2 or len(results) == 3:
        f, ax = plt.subplots(1, len(results))
        f.suptitle('Useful statistics in training')
        for i in range(len(results)):
            ax[i].plot(range(len(results[i])), results[i], '-r')
            ax[i].grid(True)
            ax[i].set_title(titles[i])

    plt.savefig('./results/' + opt.data_flag + '/' + 'losses_' + opt.model_config + '_' + opt.model_str + '.png')
    plt.show()
    plt.close()

    ########################################
    #               ACCURACY               #
    ########################################
    plt.plot(range(len(teacher_accs)), teacher_accs, '-r', label = 'teacher')
    plt.plot(range(len(student_accs)), student_accs, '--b', label = 'student')
    plt.grid(True)
    plt.title('Teacher Acc v.s. Student Acc')
    plt.legend(loc = 'best')
    plt.savefig('./results/' + opt.data_flag + '/' + 'accs_' + opt.model_config + '_' + opt.model_str + '.png')
    plt.show()
    plt.close()

def clip_sigmoid(x, clipped_value = 0.5):
    """This function is used to clip linear output
    Args :
        --x: shape[m, c] c is the channel
        --clipped_value: clipped value, default is 0.5
    return :
        --x: clipped_x
    """
    x = t.where(x < clipped_value, t.full_like(x, clipped_value), x)

    return x

def summary(model, input_size, batch_size = -1, device = "cuda"):
    """This file is modified from torchsummary.summary API"""
    def register_hook(module):
        def hook(module, input, output):
            class_name = str(module.__class__).split(".")[-1].split("'")[0]
            module_idx = len(summary)

            m_key = "%s-%i" % (class_name, module_idx + 1)
            summary[m_key] = OrderedDict()
            summary[m_key]["input_shape"] = list(input[0].size())
            summary[m_key]["input_shape"][0] = batch_size
            if isinstance(output, (list, tuple)):
                summary[m_key]["output_shape"] = [
                    [-1] + list(o.size())[1:] for o in output
                ]
            else:
                summary[m_key]["output_shape"] = list(output.size())
                summary[m_key]["output_shape"][0] = batch_size

            params = 0
            if hasattr(module, "weight") and hasattr(module.weight, "size"):
                params += t.prod(t.LongTensor(list(module.weight.size())))
                summary[m_key]["trainable"] = module.weight.requires_grad
            if hasattr(module, "bias") and hasattr(module.bias, "size"):
                params += t.prod(t.LongTensor(list(module.bias.size())))
            summary[m_key]["nb_params"] = params

        if (
            not isinstance(module, t.nn.Sequential)
            and not isinstance(module, t.nn.ModuleList)
            and not (module == model)
        ):
            hooks.append(module.register_forward_hook(hook))

    device = device.lower()
    assert device in [
        "cuda",
        "cpu",
    ], "Input device is not valid, please specify 'cuda' or 'cpu'"

    if device == "cuda" and t.cuda.is_available():
        dtype = t.cuda.FloatTensor
    else:
        dtype = t.FloatTensor

    # multiple inputs to the network
    if isinstance(input_size, tuple):
        input_size = [input_size]

    # batch_size of 2 for batchnorm
    x = [t.rand(2, *in_size).type(dtype) for in_size in input_size]
    # print(type(x[0]))

    # create properties
    summary = OrderedDict()
    hooks = []

    # register hook
    model.apply(register_hook)

    # make a forward pass
    model(*x)

    # remove these hooks
    for h in hooks:
        h.remove()

    string = '----------------------------------------------------------------\n'
    print("----------------------------------------------------------------")
    line_new = "{:>20}  {:>25} {:>15}".format("Layer (type)", "Output Shape", "Param #")
    print(line_new)
    string += line_new + '\n'
    print("================================================================")
    string += '================================================================\n'
    total_params = 0
    total_output = 0
    trainable_params = 0

    for layer in summary:
        # input_shape, output_shape, trainable, nb_params
        line_new = "{:>20}  {:>25} {:>15}".format(
            layer,
            str(summary[layer]["output_shape"]),
            "{0:,}".format(summary[layer]["nb_params"]),
        )
        total_params += summary[layer]["nb_params"]
        total_output += np.prod(summary[layer]["output_shape"])
        if "trainable" in summary[layer]:
            if summary[layer]["trainable"] == True:
                trainable_params += summary[layer]["nb_params"]
        print(line_new)
        string += line_new + '\n'

    # assume 4 bytes/number (float on cuda).
    total_input_size = abs(np.prod(input_size) * batch_size * 4. / (1024 ** 2.))
    total_output_size = abs(2. * total_output * 4. / (1024 ** 2.))  # x2 for gradients
    total_params_size = abs(total_params.numpy() * 4. / (1024 ** 2.))
    total_size = total_params_size + total_output_size + total_input_size

    print("================================================================")
    string += '================================================================\n'
    print("Total params: {0:,}".format(total_params))
    string += "Total params: {0:,}".format(total_params) + '\n'
    print("Trainable params: {0:,}".format(trainable_params))
    string += "Trainable params: {0:,}".format(trainable_params) + '\n'
    print("Non-trainable params: {0:,}".format(total_params - trainable_params))
    string += "Non-trainable params: {0:,}".format(total_params - trainable_params) + '\n'
    print("----------------------------------------------------------------")
    string += '----------------------------------------------------------------\n'
    print("Input size (MB): %0.2f" % total_input_size)
    string += "Input size (MB): %0.2f" % total_input_size + '\n'
    print("Forward/backward pass size (MB): %0.2f" % total_output_size)
    string += "Forward/backward pass size (MB): %0.2f" % total_output_size + '\n'
    print("Params size (MB): %0.2f" % total_params_size)
    string += "Params size (MB): %0.2f" % total_params_size + '\n'
    print("Estimated Total Size (MB): %0.2f" % total_size)
    string += "Estimated Total Size (MB): %0.2f" % total_size + '\n'
    print("----------------------------------------------------------------")
    string += '----------------------------------------------------------------\n'

    return string

def preprocess_tiny_imagenet():
    """This function is used to preprocess tiny imagenet dataset
    The downloaded tiny imagenet has bounding boxes, which we will not use in this paper
    we first extract images from 'images' folder from different classes, then we create file tree like this:
    train:
        |-class_name1
            |-image1
            |-image2
            ...
        |-class_name2
            |-image1
            |-image2
            ...
        ...
    val:
        |-images
        |-labels
    which is already preprocessed for us, so we don't have to have any modifications
    in original tiny imagenet data set, we have no access to test labels, so we use val as test result and
    report our results on val(test) data set
    """
    data_path = './data/tiny_imagenet'
    train_files = os.listdir(os.path.join(data_path, 'train'))
    train_files = [data_path + os.sep + 'train' + os.sep + filename for filename in train_files]
    train_files = list(filter(lambda x : os.path.isdir(x), train_files))

    for i, train_file in enumerate(train_files):
        files = os.listdir(train_file + os.sep + 'images')
        files = [train_file + os.sep + 'images' + os.sep + file for file in files]
        files = list(filter(lambda x : x.endswith('JPEG'), files))
        print('\nPreprocessing %d / %d folder : %s.' % (i + 1, len(train_files), train_file))
        for j, file in enumerate(files):
            sys.stdout.write('\r\tMoving %d / %d file %s for this class.' % (j + 1, len(files), file))
            sys.stdout.flush()
            move(file, os.sep.join(file.split(os.sep)[:-2]) + os.sep + ''.join(file.split(os.sep)[-1]))
            #copyfile(file, os.sep.join(file.split(os.sep)[:-2]) + os.sep + ''.join(file.split(os.sep)[-1]))

        rmtree(train_file + os.sep + 'images')
        del_file = [file for file in os.listdir(train_file) if file.endswith('txt')][0]
        os.remove(train_file + os.sep + del_file)

    print('\ndone!')

def get_tiny_imagenet_val_labels(images_path):
    """This function is used to get tiny_imagenet val labels
    Args :
        --images_path: whole val folder absolute path
    return :
        --labels: Python list
    """
    par_path = os.sep.join(images_path.split(os.sep)[:-1])
    class2idx = joblib.load(os.path.join(par_path, 'class2idx.pkl'))

    if not os.path.exists(images_path + os.sep + 'val_labels.txt'):
        labels_file = os.path.join(images_path, 'val_annotations.txt')
        with open(labels_file, 'r') as f:
            labels = []
            for line in f:
                line = line.strip('\n').split()[1]
                labels.append(line)

        with open(images_path + os.sep + 'val_labels.txt', 'w') as f:
            strings = '\n'.join(labels)
            f.writelines(strings)
            f.flush()

    labels = []
    with open(images_path + os.sep + 'val_labels.txt', 'r') as f:
        for line in f:
            line = line.strip('\n')
            labels.append(int(class2idx[line]))

    return labels

class Dataset(t.utils.data.Dataset):
    def __init__(self, images_path, extension_format, labels = None):
        """
        Args :
            --images_path: whole images folder absolute path
            --extension_format: images format
        """
        super(Dataset, self).__init__()

        images_paths = os.listdir(images_path)
        self.images_path = images_path
        self.images_paths = list(filter(lambda x : x.endswith(extension_format), images_paths))
        self.imgs = [os.path.join(images_path, path) for path in self.images_paths]
        self.labels = labels

        if images_path.count('imagenet'):
            self.transform = tv.transforms.Compose([tv.transforms.ToTensor(),
                                                    tv.transforms.Normalize(mean = (.485, .456, .406),
                                                                            std = (.229, .224, .225))])

    def __getitem__(self, index):
        img_path = self.imgs[index]
        if self.images_path.count('imagenet'):
            label = self.labels[index]
        else:
            label = None

        pil_img = Image.open(img_path).convert('RGB')
        array = np.asarray(pil_img)
        #print(array.shape)
        #plt.show(array.astype('uint8'))
        #plt.show()
        #input()
        #data = t.from_numpy(array)
        data = self.transform(array)

        return data, label

    def __len__(self):

        return len(self.imgs)