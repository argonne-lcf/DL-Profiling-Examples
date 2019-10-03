import torch
import torchvision
import numpy
import time

def main():
  train_set = torchvision.datasets.MNIST('./', train=True, download=True,
                                        transform = torchvision.transforms.ToTensor())
  test_set = torchvision.datasets.MNIST('./', train=False, download=True,
                                        transform = torchvision.transforms.ToTensor())

  # We can get directly at the tensors:
  x_test = test_set.data.reshape([10000,1,28,28]).type(torch.FloatTensor)
  y_test = test_set.targets

  x_train = train_set.data.reshape([60000,1,28,28]).type(torch.FloatTensor)
  y_train = train_set.targets

  class MyModel(torch.nn.Module):
    def __init__(self):
      super(MyModel, self).__init__()
      self.conv1 = torch.nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3)
      self.pool1 = torch.nn.MaxPool2d(kernel_size=2)
      self.conv2 = torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3)
      # Without padding, this will end up as a shape of [C=32,W=26,H=26] for each image
      self.d1    = torch.nn.Linear(in_features=64*11*11, out_features=128)
      self.dropout = torch.nn.Dropout(0.2)
      self.d2    = torch.nn.Linear(in_features=128, out_features=10)

    @profile
    def forward(self, x):
      x = self.conv1(x)
      x = torch.relu(x)

      x = self.pool1(x)

      x = self.conv2(x)
      x = torch.relu(x)

      # Flatten will flatten arbitrary tensors, so specify from the channel index to the end:
      x = torch.flatten(x, 1, -1)
      x = self.d1(x)
      x = torch.relu(x)
      x = self.dropout(x)
      x = self.d2(x)
      return x

  # Create an instance of the model
  model = MyModel()

  # Use a list of indexes to shuffle the dataset each epoch
  indexes = numpy.arange(len(train_set))

  epochs = 1
  batch_size = 128

  # Create an instance of an optimizer:
  optimizer=torch.optim.Adam(model.parameters())

  loss_operation = torch.nn.CrossEntropyLoss()


  for epoch in range(epochs):
    start = time.time()
    # Shuffle the indexes:
    numpy.random.shuffle(indexes)

    @profile
    for batch in range(len(indexes/batch_size)):
      if (batch+1)*batch_size > 60000:
        continue

      batch_indexes = indexes[batch*batch_size:(batch+1)*batch_size]
      images = x_train[batch_indexes]
      labels = y_train[batch_indexes].reshape([batch_size,])

      # Set the model to training mode:
      model.train()
      # Reset the gradient values for this step:
      optimizer.zero_grad()
      # Compute the logits:
      logits = model(images)


      # Loss value is computed imperatively
      loss = loss_operation(input=logits, target=labels)
      # This call performs the back prop:
      loss.backward()
      # This call updates the weights using the optimizer
      optimizer.step()

    end = time.time()

    with torch.no_grad():
      model.eval()

      #Evaluate the accuracy on the test set:
      test_logits = model(x_test)
      test_loss = loss_operation(input=test_logits, target=y_test)

      accuracy = numpy.mean(numpy.argmax(test_logits.numpy(), axis=-1) == y_test.numpy())

      print("Accuracy after epoch {} is:".format(epoch), accuracy, " {:.2f} s".format(end-start))

if __name__ == '__main__':
  main()
