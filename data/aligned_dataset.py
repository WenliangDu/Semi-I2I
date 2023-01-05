import os.path
import scipy.io as sio # for reading .mat file (segmentation)
#import random
from data.base_dataset import BaseDataset, get_params, get_transform
import torchvision.transforms as transforms
from data.image_folder import make_dataset
from PIL import Image
import numpy as np
from util import util


class AlignedDataset(BaseDataset):
    """A dataset class for paired image dataset.

    It assumes that the directory '/path/to/data/train' contains image pairs in the form of {A,B}.
    During test time, you need to prepare a directory '/path/to/data/test'.
    """

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        self.dir_A = os.path.join(opt.dataroot, opt.phase + '_A')  # create a path '/path/to/data/trainA'
        self.dir_B = os.path.join(opt.dataroot, opt.phase + '_B')  # create a path '/path/to/data/trainB'

        self.A_paths = sorted(make_dataset(self.dir_A, opt.max_dataset_size))   # load images from '/path/to/data/trainA'
        self.B_paths = sorted(make_dataset(self.dir_B, opt.max_dataset_size))    # load images from '/path/to/data/trainB'
        self.A_size = len(self.A_paths)  # get the size of dataset A
        self.B_size = len(self.B_paths)  # get the size of dataset B
        btoA = self.opt.direction == 'BtoA'
        self.input_nc = self.opt.output_nc if btoA else self.opt.input_nc       # get the number of channels of input image
        self.output_nc = self.opt.input_nc if btoA else self.opt.output_nc      # get the number of channels of output image


    def __getitem__(self, index):

        """Return a data point and its metadata information.

                Parameters:
                    index (int)      -- a random integer for data indexing

                Returns a dictionary that contains A, B, A_paths and B_paths
                    A (tensor)       -- an image in the input domain
                    B (tensor)       -- its corresponding image in the target domain
                    A_paths (str)    -- image paths
                    B_paths (str)    -- image paths
        """
        # apply the same transform to A
        A_path = self.A_paths[index]
        A_img = Image.open(A_path).convert('RGB')
        transform_params = get_params(self.opt, A_img.size)
        #A_transform, A_transform_NoTenNorm = get_transform(self.opt, transform_params, grayscale=(self.input_nc == 1))
        A_transform = get_transform(self.opt, transform_params, grayscale=(self.input_nc == 1))
        A = A_transform(A_img)

        # apply the same transform to B
        B_path = self.B_paths[index]
        B_img = Image.open(B_path).convert('RGB')
        B_transform = get_transform(self.opt, transform_params, grayscale=(self.output_nc == 1)) #21/05/04
        #B_transform, B_transform_NoTenNorm = get_transform(self.opt, transform_params, grayscale=(self.output_nc == 1))
        B = B_transform(B_img)

        return {'A': A, 'B': B, 'A_paths': A_path, 'B_paths': B_path}

    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.A_paths)
