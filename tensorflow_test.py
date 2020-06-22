'''
@Author: miaoyu
@Date: 2020-06-19 17:17:18
@LastEditTime: 2020-06-19 20:08:47
@LastEditors: miaoyu
@Description: 
'''
import tensorflow as tf
import numpy as np

# create data
x_data = np.random.rand(100).astype(np.float32)
y_data = x_data * 0.1 + 0.3

# print(dir(tf))

### create tensorflow structrue start ###
Weights = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
biases = tf.Variable(tf.zeros([1]))

y = Weights*x_data + biases
loss = tf.reduce_mean(tf.square(y - y_data))

optimizer = tf.train.GradientDescentOptimizer(0.5)
train = optimizer.minimize(loss)

init = tf.initialize_all_varibles()
### create tensorflow structrue end ###

sess = tf.Session()

sess.run(init)

for step in range(201):
  sess.run(train)

  if step % 20 == 0:
    print(step, sess.run(Weights), sess.run(biases))
