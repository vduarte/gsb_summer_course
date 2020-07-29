import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
import matplotlib.pyplot as plt

batch_size = 10000
optimizer = tf.optimizers.Adam()


def f(x):
  return x**2


net = Sequential([Dense(64, 'relu', input_dim=1),
                  Dense(32, 'relu'),
                  Dense(1)])

net.summary()
Θ = net.weights


@tf.function
def train_step():
    x = tf.random.uniform(shape=[batch_size, 1])
    with tf.GradientTape() as tape:
      y_hat = net(x)
      loss = tf.reduce_mean((y_hat - f(x))**2)

    grad = tape.gradient(loss, Θ)
    optimizer.apply_gradients(zip(grad, Θ))
    return loss


for i in range(1000):
  train_step()
  if i % 100 == 0:
    print(train_step())


x = np.linspace(0, 1, 1000).reshape(-1, 1)
plt.plot(x, f(x))
plt.plot(x, net(x))
