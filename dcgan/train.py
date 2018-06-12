import time
from tqdm import tqdm

from dcgan.model import *
from config import *
from utils import *


def dcgan_train():
    print('Preparing dara')
    # prepare images for train
    train_hr_img_list = sorted(tl.files.load_file_list(train_hr_img_path, regx='.*.png', printable=False))
    train_hr_images = tl.visualize.read_images(train_hr_img_list, train_hr_img_path, printable=False)

    # input to generator
    t_image = tf.placeholder(tf.float32, [batch_size, 96, 96, 3])
    # train target
    t_target_image = tf.placeholder(tf.float32, [batch_size, 384, 384, 3])

    # ========== define model ==========
    gen = dcgan_generator(t_image, is_train=True)
    dis, logits_real = dcgan_discriminator(t_target_image, is_train=True)
    _, logits_fake = dcgan_discriminator(gen.outputs, is_train=True, reuse=True)

    # ========== losses ==========
    d_loss1 = 0.9 * tl.cost.absolute_difference_error(logits_real, tf.ones_like(logits_real))
    d_loss2 = 0.1 * tl.cost.sigmoid_cross_entropy(logits_fake, tf.zeros_like(logits_fake))
    d_loss = d_loss1 + d_loss2

    g_loss = tl.cost.sigmoid_cross_entropy(logits_fake, tf.ones_like(logits_fake))

    # ========== optimizer ==========
    g_vars = tl.layers.get_variables_with_name('dcgan_gen')
    d_vars = tl.layers.get_variables_with_name('dcgan_dis')
    g_opt = tf.train.AdamOptimizer(learning_rate).minimize(g_loss, var_list=g_vars)
    d_opt = tf.train.AdamOptimizer(learning_rate).minimize(d_loss, var_list=d_vars)

    sess = tf.Session()
    tl.layers.initialize_global_variables(sess)

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
    tl.files.save_npz(gen.all_params, name=checkpoint_dir + '/generator_dcgan.npz', sess=sess)
    tl.files.save_npz(dis.all_params, name=checkpoint_dir + '/discriminator_dcgan.npz', sess=sess)

    print('time: ', time.time() - start_time)
