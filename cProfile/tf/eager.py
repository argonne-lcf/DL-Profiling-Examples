import tensorflow as tf
import numpy
import time

# Enable eager execution
tf.enable_eager_execution()

# Load the dataset and cast to the right formats:
mnist = tf.keras.datasets.mnist

(x_train, y_train),(x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

x_train, x_test = x_train.reshape([60000,28,28,1]), x_test.reshape([10000,28,28,1])
y_train, y_test = y_train.astype(numpy.int32), y_test.astype(numpy.int32)

class MyModel(tf.keras.Model):
  def __init__(self):
    super(MyModel, self).__init__()
    self.conv1      = tf.keras.layers.Conv2D(32, 3, activation='relu')
    self.pool       = tf.keras.layers.MaxPooling2D()
    self.conv2      = tf.keras.layers.Conv2D(32, 3, activation='relu')
    self.flatten    = tf.keras.layers.Flatten()
    self.d1         = tf.keras.layers.Dense(128, activation='relu')
    self.dropout    = tf.keras.layers.Dropout(0.2)
    self.d2         = tf.keras.layers.Dense(10, activation=None)

  def call(self, x):

    x = self.conv1(x)
    x = self.pool(x)
    x = self.conv2(x)
    x = self.flatten(x)
    x = self.d1(x)
    x = self.dropout(x)
    x = self.d2(x)

    return x

# Create an instance of the model
model = MyModel()

# Use a list of indexes to shuffle the dataset each epoch
indexes = numpy.arange(len(x_train))

epochs = 5
batch_size = 64

# Create an instance of an optimizer:
optimizer=tf.train.AdamOptimizer()

for epoch in range(5):
  # Shuffle the indexes:
  numpy.random.shuffle(indexes)

  for batch in range(len(indexes/batch_size)):

    batch_indexes = indexes[batch*batch_size:(batch+1)*batch_size]

    images = x_train[batch_indexes]
    labels = y_train[batch_indexes]
    print(labels.shape)
    labels = labels.reshape([batch_size,])

    # Gradient tape indicates to TF to build a graph on the fly.
    with tf.GradientTape() as tape:
      # This line is the forward pass of the network:
      # (The first call to model will initialize the weights)
      logits = model(images)
      # Loss value is computed imperatively
      loss_value = tf.losses.sparse_softmax_cross_entropy(logits=logits, labels=labels)

    # Compute the backward pass with the gradient tape:
    grads = tape.gradient(loss_value, model.trainable_variables)
    # Use the optimizer to update the gradients:
    optimizer.apply_gradients(zip(grads, model.trainable_variables))


  #Evaluate the accuracy on the test set:
  test_logits = model(x_test).numpy()
  accuracy = numpy.mean(numpy.argmax(test_logits, axis=-1) == y_test)

  print("Accuracy after epoch {} is:".format(epoch), accuracy)
