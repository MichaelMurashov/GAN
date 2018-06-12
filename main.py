import tensorlayer as tl
import os
import sys

from srgan.train import srgan_train
from srgan.valid import srgan_valid
from dcgan.train import dcgan_train
from dcgan.valid import dcgan_valid
from config import *


if __name__ == '__main__':
    tl.files.exists_or_mkdir(samples_valid_dir_srgan)
    tl.files.exists_or_mkdir(samples_valid_dir_dcgan)
    tl.files.exists_or_mkdir(checkpoint_dir)

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    if len(sys.argv) > 1:
        if sys.argv[1] == 'srgan':
            if sys.argv[2] == 'train':
                srgan_train()
            elif sys.argv[2] == 'valid':
                srgan_valid(checkpoint_dir + 'generator_srgan.npz')
            else:
                print('invalid param')

        elif sys.argv[1] == 'dcgan':
            if sys.argv[2] == 'train':
                dcgan_train()
            elif sys.argv[2] == 'valid':
                dcgan_valid(checkpoint_dir + 'generator_dcgan.npz')
            else:
                print('invalid param')

        else:
            print('invalid param')
