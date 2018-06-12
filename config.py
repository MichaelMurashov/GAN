batch_size = 5
learning_rate = 1e-4
max_epoch = 200

train_hr_img_path = 'data/DIV2K_train_HR/'

valid_hr_img_path = 'data/DIV2K_valid_HR/'
valid_lr_img_path = 'data/DIV2K_valid_LR_bicubic/X4/'

checkpoint_dir = 'checkpoint/'
samples_valid_dir_srgan = 'samples/srgan/'
samples_valid_dir_dcgan = 'samples/dcgan/'

vgg19_npy_path = 'srgan/vgg19/vgg19.npy'
