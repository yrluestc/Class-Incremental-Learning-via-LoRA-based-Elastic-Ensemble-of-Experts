import logging
try:
    import requests
except ImportError as e:
    logging.error("Please install requests using 'pip install requests'")
    raise e

import os
import torchvision.transforms as transforms
import torchvision.datasets as torchvision_datasets
import torch.nn.functional as F
from torch.utils.data import Dataset
import numpy as np
import pickle
from PIL import Image
from typing import Tuple

import yaml

from datasets.utils import set_default_from_args
from utils import smart_joint
from utils.conf import base_path
from datasets.utils.continual_dataset import ContinualDataset, fix_class_names_order, store_masked_loaders, getAllLoaders
from datasets.transforms.denormalization import DeNormalize
from torchvision.transforms.functional import InterpolationMode


def split_images_labels(imgs):
    # split trainset.imgs in ImageFolder
    images = []
    labels = []
    for item in imgs:
        images.append(item[0])
        labels.append(item[1])

    return np.array(images), np.array(labels)

class MyDomainNet(Dataset):
    N_CLASSES = 69 * 5

    """
    Overrides the CIFAR100 dataset to change the getitem function.
    """

    def __init__(self, root, train=True, transform=None,
                 target_transform=None, download=False) -> None:

        self.root = root
        self.train = train
        self.transform = transform
        self.target_transform = target_transform

        self.not_aug_transform = transforms.Compose([transforms.Resize((224, 224), interpolation=InterpolationMode.BICUBIC), transforms.ToTensor()])

        self.domain_names = ["clipart", "infograph", "painting", "quickdraw", "real", "sketch", ]

        # #complete
        # if self.train:
        #     image_list_paths = [os.path.join(self.root, d + '_train.txt') for d in self.domain_names]
        #     imgs = []
        #     for i in range(len(self.domain_names)):
        #         img_path = image_list_paths[i]
        #         image_list = open(img_path).readlines()
        #         imgs += [(val.split()[0], int(val.split()[1]) + i * 69) for val in image_list]
        #     train_x = []
        #     train_y = []
        #     for item in imgs:
        #         train_x.append(os.path.join(self.root, item[0]))
        #         train_y.append(item[1])
        #     self.data = np.array(train_x)
        #     self.targets = np.array(train_y)
        # else:
        #     image_list_paths = [os.path.join(self.root, d + '_test.txt') for d in self.domain_names]
        #     imgs = []
        #     for i in range(len(self.domain_names)):
        #         img_path = image_list_paths[i]
        #         image_list = open(img_path).readlines()
        #         imgs += [(val.split()[0], int(val.split()[1]) + i * 69) for val in image_list]
        #     test_x = []
        #     test_y = []
        #     for item in imgs:
        #         test_x.append(os.path.join(self.root, item[0]))
        #         test_y.append(item[1])
        #     self.data = np.array(test_x)
        #     self.targets = np.array(test_y)

        #345classes
        if self.train:
            train_data_config = yaml.load(open('datasets/domainnet_train.yaml', 'r'), Loader=yaml.Loader)
            self.data = np.array(train_data_config['data'])
            self.targets = np.array(train_data_config['targets'])
        else:
            test_data_config = yaml.load(open('datasets/domainnet_test.yaml', 'r'), Loader=yaml.Loader)
            self.data = np.array(test_data_config['data'])
            self.targets = np.array(test_data_config['targets'])

    def __len__(self):
        return len(self.targets)#可能有问题

    def __getitem__(self, index: int) -> Tuple[Image.Image, int, Image.Image]:
        """
        Gets the requested element from the dataset.
        :param index: index of the element to be returned
        :returns: tuple: (image, target) where target is index of the target class.
        """

        img, target = self.data[index], self.targets[index]

        img = Image.open(img).convert('RGB')

        original_img = img.copy()

        not_aug_img = self.not_aug_transform(original_img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        if not self.train:
            return img, target

        if hasattr(self, 'logits'):
            return img, target, not_aug_img, self.logits[index]

        return img, target, not_aug_img


class SequentialDomainNet(ContinualDataset):

    NAME = 'seq-domainnet'
    SETTING = 'class-il'
    N_TASKS = 5
    N_CLASSES = 69 * 5
    N_CLASSES_PER_TASK = N_CLASSES // N_TASKS
    MEAN, STD = (0.0, 0.0, 0.0), (1.0, 1.0, 1.0)
    SIZE = (224, 224)

    TRANSFORM = transforms.Compose([
        transforms.RandomResizedCrop(SIZE[0], interpolation=InterpolationMode.BICUBIC),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD),
    ])
    TEST_TRANSFORM = transforms.Compose([transforms.Resize(size=(256, 256),
                                                           interpolation=InterpolationMode.BICUBIC),
                                         transforms.CenterCrop(SIZE[0]),
                                         transforms.ToTensor(),
                                         transforms.Normalize(mean=MEAN, std=STD)])

    def get_data_loaders(self):
        train_dataset = MyDomainNet(base_path() + 'domainnet/', train=True,
                                    download=True, transform=self.TRANSFORM)

        test_dataset = MyDomainNet(base_path() + 'domainnet/', train=False,
                                   download=True, transform=self.TEST_TRANSFORM)

        train, test = store_masked_loaders(train_dataset, test_dataset, self)
        return train, test

    def get_all_data_loaders(self):
        """Class method that returns the train and test loaders."""
        train_dataset = MyDomainNet(base_path() + 'domainnet/', train=True,
                                  download=True, transform=self.TRANSFORM)
        test_dataset = MyDomainNet(base_path() + 'domainnet/', train=False,
                                download=True, transform=self.TEST_TRANSFORM)

        train, test = getAllLoaders(train_dataset, test_dataset, self)
        return train, test

    # def get_class_names(self): #需后续实现
    #     if self.class_names is not None:
    #         return self.class_names
    #
    #     pwd = os.path.dirname(os.path.abspath(__file__))
    #     with open(pwd + '/imagenet_r_utils/label_to_class_name.pkl', 'rb') as f:
    #         label_to_class_name = pickle.load(f)
    #     class_names = label_to_class_name.values()
    #     class_names = [x.replace('_', ' ') for x in class_names]
    #
    #     class_names = fix_class_names_order(class_names, self.args)
    #     self.class_names = class_names
    #     return self.class_names

    @staticmethod
    def get_transform():
        transform = transforms.Compose(
            [transforms.ToPILImage(), SequentialDomainNet.TRANSFORM])
        return transform

    @set_default_from_args("backbone")
    def get_backbone():
        return "vit"

    @staticmethod
    def get_loss():
        return F.cross_entropy

    @staticmethod
    def get_normalization_transform():
        return transforms.Normalize(mean=SequentialDomainNet.MEAN, std=SequentialDomainNet.STD)

    @staticmethod
    def get_denormalization_transform():
        transform = DeNormalize(SequentialDomainNet.MEAN, SequentialDomainNet.STD)
        return transform

    @set_default_from_args('n_epochs')
    def get_epochs(self):
        return 50

    @set_default_from_args('batch_size')
    def get_batch_size(self):
        return 128
