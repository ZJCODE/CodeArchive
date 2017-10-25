from dcgan import DCGAN
from sklearn.utils import shuffle
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


N_EPOCH = 10
BATCH_SIZE = 32
G_SIZE = 100


def gen_batch(arr, batch_size):
    for i in range(0, len(arr), batch_size):
        yield arr[i : i + batch_size]


def scaled(images):
    return ( images.astype(np.float32) - (255./2) ) / (255./2)


if __name__ == '__main__':
    (X_train, y_train), (_, _) = tf.contrib.keras.datasets.mnist.load_data()
    X_train = scaled(X_train)
    X = np.expand_dims(X_train, 3)[y_train == 8]
    
    gan = DCGAN(G_SIZE)
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    
    plt.figure()
    for i, epoch in enumerate(range(N_EPOCH)):
        X = shuffle(X)
        for step, images in enumerate(gen_batch(X, BATCH_SIZE)):
            noise = np.random.randn(len(images), G_SIZE)

            sess.run(gan.D_train, {gan.G_in: noise, gan.X_in: images, gan.train_flag: True})
            for _ in range(2):
                sess.run(gan.G_train, {gan.G_in: noise, gan.train_flag: True})

            G_loss, D_loss, D_prob, G_prob, mse = sess.run([gan.G_loss, gan.D_loss, gan.X_prob, gan.G_prob, gan.mse],
                                                           {gan.G_in: noise, gan.X_in: images, gan.train_flag: False})
            print("Epoch %d/%d | Step %d/%d" % (epoch+1, N_EPOCH, step, len(X)//BATCH_SIZE))
            print("G loss: %.4f | D loss: %.4f | D prob: %.4f | G prob: %.4f | mse: %.4f" %
                 (G_loss, D_loss, D_prob.mean(), G_prob.mean(), mse))
        
        if i in range(N_EPOCH-4, N_EPOCH):
            img = sess.run(gan.G_out, {gan.G_in: noise, gan.train_flag: False})[0]
            plt.subplot(2, 2, i+1-(N_EPOCH-4))
            plt.imshow(np.squeeze(img))
    plt.tight_layout()
    plt.show()
    