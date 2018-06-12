import time
from tqdm import tqdm
import numpy as np

from config import *
from srgan.model import *
from utils import *
from srgan.vgg19.vgg19_api import get_vgg


def srgan_train():
    print('Preparing dara')
    # prepare images for train
    train_hr_img_list = sorted(tl.files.load_file_list(train_hr_img_path, regx='.*.png', printable=False))
    train_hr_images = tl.visualize.read_images(train_hr_img_list, train_hr_img_path, printable=False)

    # input to generator
    t_image = tf.placeholder(tf.float32, [batch_size, 96, 96, 3])
    # train target
    t_target_image = tf.placeholder(tf.float32, [batch_size, 384, 384, 3])

    # ========== define model ==========
    gen = srgan_generator(t_image, is_train=True)
    dis, logits_real = srgan_discriminator(t_target_image, is_train=True)
    _, logits_fake = srgan_discriminator(gen.outputs, is_train=True, reuse=True)

    # ========== load vgg ==========
    npz = np.load(vgg19_npy_path, encoding='latin1').item()
    net_vgg, vgg_loss, params = get_vgg(t_target_image, gen.outputs, npz)

    # ========== losses ==========
    d_loss1 = tl.cost.sigmoid_cross_entropy(logits_real, tf.ones_like(logits_real), name='d1')
    d_loss2 = tl.cost.sigmoid_cross_entropy(logits_fake, tf.zeros_like(logits_fake), name='d2')
    d_loss = d_loss1 + d_loss2

    g_gan_loss = 1e-3 * tl.cost.sigmoid_cross_entropy(logits_fake, tf.ones_like(logits_fake))
    g_loss = vgg_loss + g_gan_loss

    # ========== optimizer ==========
    g_vars = tl.layers.get_variables_with_name('srgan_gen')
    d_vars = tl.layers.get_variables_with_name('srgan_dis')
    g_opt = tf.train.AdamOptimizer(learning_rate).minimize(g_loss, var_list=g_vars)
    d_opt = tf.train.AdamOptimizer(learning_rate).minimize(d_loss, var_list=d_vars)

    sess = tf.Session()
    tl.layers.initialize_global_variables(sess)
    tl.files.assign_params(sess, params, net_vgg)

    # ========== train ==========
    start_time = time.time()
    for epoch in range(1, max_epoch + 1, 1):
        print('[*] epoch: ', epoch)

        for idx in tqdm(range(0, len(train_hr_images), batch_size)):
            b_images_384 = tl.prepro.threading_data(train_hr_images[idx:idx + batch_size],
                                                    fn=crop_sub_images, is_random=True)
            b_images_96 = tl.prepro.threading_data(b_images_384, fn=downsample)

            # train dis
            sess.run([d_loss, d_opt], {t_image: b_images_96, t_target_image: b_images_384})
            # train gen
            sess.run([g_loss, g_opt], {t_image: b_images_96, t_target_image: b_images_384})

    # save model
    tl.files.save_npz(gen.all_params, name=checkpoint_dir + '/generator_srgan.npz', sess=sess)
    tl.files.save_npz(dis.all_params, name=checkpoint_dir + '/discriminator_srgan.npz', sess=sess)

    print('time: ', time.time() - start_time)
