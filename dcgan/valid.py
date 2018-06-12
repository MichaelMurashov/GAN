import time
import scipy

from dcgan.model import *
from config import *


def dcgan_valid(path):
    valid_hr_img_list = sorted(tl.files.load_file_list(valid_hr_img_path, regx='.*.png', printable=False))
    valid_lr_img_list = sorted(tl.files.load_file_list(valid_lr_img_path, regx='.*.png', printable=False))

    valid_hr_images = tl.visualize.read_images(valid_hr_img_list, valid_hr_img_path, printable=False)
    valid_lr_images = tl.visualize.read_images(valid_lr_img_list, valid_lr_img_path, printable=False)

    image_id = 8
    valid_hr_img = valid_hr_images[image_id]
    valid_lr_img = valid_lr_images[image_id]

    valid_lr_img = (valid_lr_img / 127.5) - 1

    size = valid_lr_img.shape

    t_image = tf.placeholder(tf.float32, [1, None, None, 3])

    gen = dcgan_generator(t_image, is_train=False)

    sess = tf.Session()
    tl.layers.initialize_global_variables(sess)
    tl.files.load_and_assign_npz(sess, path, gen)

    start_time = time.time()
    out = sess.run(gen.outputs, {t_image: [valid_lr_img]})
    print('time: ', time.time() - start_time)

    tl.visualize.save_image(out[0], samples_valid_dir_dcgan + 'gen.png')
    tl.visualize.save_image(valid_lr_img, samples_valid_dir_dcgan + 'lr.png')
    tl.visualize.save_image(valid_hr_img, samples_valid_dir_dcgan + 'hr.png')

    bicub = scipy.misc.imresize(valid_lr_img, [size[0]*4, size[1]*4], interp='bicubic', mode=None)
    tl.visualize.save_image(bicub, samples_valid_dir_dcgan + 'bicubic.png')
