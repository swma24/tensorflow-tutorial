from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

# download data
# mnist is traning/validation/test set as Numpy array
# Also it provides a function for iterating through data minibatches
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

# InteractiveSession을 쓰지 않으면 세션에 그래프를 올리기 전에 그래프를 전부 완성해야 함.
# 다른말로 하면 계산하기 전에 그래프를 완성해야 함. InteractiveSession을 쓰면 그때 그때 계산이 가능함.
sess = tf.InteractiveSession()

# 1-layer NN => Softmax. 1-layer란 Input layer와 Output layer만 있는 것이 1-layer임.
# placeholder는 실행할때 우리가 값을 넣을 수 있음
x = tf.placeholder(tf.float32, shape=[None, 784])  # x는 input image. 784 = 28*28, 이미지를 핌 (flatten). 흑백 이미지이므로 각 픽셀은 0/1
y_ = tf.placeholder(tf.float32, shape=[None, 10])  # _y는 class label. mnist가 0~9까지의 이미지이므로 10개. one-hot 벡터.

# Variables: Weights & Bias
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

# 변수는 사용하기 전에 초기화해줘야 한다. 선언시에 정해준 값으로 (여기서는 0) 초기화된다.
tf.initialize_all_variables().run()

# Feed-forward 수행
y = tf.matmul(x, W) + b
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y, y_))  # reduce는 텐서를 축소한다는 개념인 듯.
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

# 100 크기의 mini-batch로 1000번 학습을 함.
for i in range(1000):
    batch = mnist.train.next_batch(100)
    train_step.run(feed_dict={x: batch[0], y_: batch[1]})

    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print(accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels}))

# test accuracy: 0.9179
