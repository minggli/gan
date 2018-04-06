import tensorflow as tf

from models.official.mnist.dataset import train, test
from config import BATCH_SIZE
sess = tf.Session()

mnist_train = train("./mnist_data/")
mnist_test = test("./mnist_data/")
mnist = mnist_train.concatenate(mnist_test)
mnist_batch = mnist.shuffle(1000).repeat().batch(BATCH_SIZE)
mnist_batch_iter = mnist_batch.make_initializable_iterator()

#
# def fetch_batch(Iterator):
#     with tf.Session() as sess:
#         sess.run(tf.global_variables_initializer())
#         data, labels = Iterator.get_next()
#     return data.eval(session=tf.Session())
#

with sess:
    init = mnist_batch_iter.initializer
    sess.run(init)
    data, labels = mnist_batch_iter
    print(data.eval(session=sess))
