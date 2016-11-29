from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import time


# truncated normal distribution에 기반해서 랜덤한 값으로 초기화
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


# x (아래 함수들에서) : A 4-D `Tensor` with shape `[batch, height, width, channels]`
def conv2d(x, W):
    return tf.nn.conv2d(x, filter=W, strides=[1, 1, 1, 1], padding='SAME')
    # filter: [filter_height, filter_width, in_channels, out_channels]

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    # ksize: The size of the window for each dimension of the input tensor


mnist = input_data.read_data_sets('data', one_hot=True)

sess = tf.InteractiveSession()

x = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, shape=[None, 10])

# First Convolutional Layer

# [5, 5, 1, 32]: 5x5 convolution filter, 1 input channel, 32 output channel.
# MNIST의 pixel은 gray scale로 표현되는 1개의 벡터이므로 1 input channel임.
W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])
# 최종적으로, 32개의 output channel에 대해 각각 5x5의 convolution filter weight 와 1개의 bias 를 갖게 됨.

# x는 [None, 784] (위 placeholder에서 선언, 784 = 28*28).
# x_image는 [batch, in_height, in_width, output_channels] 이 됨. -1은 batch size를 유지하는 것.
x_image = tf.reshape(x, [-1, 28, 28, 1])

# 이제, x_image를 weight tensor(filter)와 convolve하고 bias를 더한 뒤 ReLU를 적용. 그리고 마지막으론 max pooling.
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

# Second Convolutional Layer

# [5, 5, 1, 32]: 5x5 convolution filter, 32 input channel, 64 output channel.
W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

# Fully-Connected Layer

# max pooling 2번 (28 -> 14 -> 7)
# 7*7*64는 h_pool2의 output (7*7의 reduced image * 64개의 채널). 1024는 fc layer의 뉴런 수.
W_fc1 = weight_variable([7*7*64, 1024])
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])  # -1은 batch size를 유지하는 것.
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

# Dropout. training 동안만 적용하고 testing 때는 적용하지 않는다.
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# Readout Layer

W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])

y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y_conv, y_))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
sess.run(tf.initialize_all_variables())
start_time = time.time()
for i in range(20000):
    batch = mnist.train.next_batch(50)
    if i % 100 == 0:
        train_accuracy = accuracy.eval(feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0})  # keen_prob = 1.0 로 dropout 미적용.
        print("step %d, training accuracy %g" % (i, train_accuracy))
    train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})  # keen_prob = 0.5 로 dropout 50% 적용.
print("total training time is %g seconds" %(time.time() - start_time))
print("test accuracy %g" % accuracy.eval(feed_dict={x: mnist.test.images[:5000], y_: mnist.test.labels[:5000], keep_prob: 1.0}))
# test accuracy 계산 시 메모리 부족 에러가 생겨서 테스트 이미지를 50%인 5000개만 활용
# test accuracy: 0.9872
