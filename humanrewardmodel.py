import tensorflow as tf
import pickle
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.python.keras import Model, layers

with open("./data.pkl",'rb') as f:
    data=pickle.load(f)

# MNIST dataset parameters.
num_classes = 10 
num_features = 30 # data features 

# Training parameters.
learning_rate = 0.001
learning_rate_decay = 0.99
regulation_rate = 0.0001
training_steps = 2000
batch_size = 200
display_step = 100

# Network parameters.
n_hidden_1 = 128 # 1st layer number of neurons.
n_hidden_2 = 64 # 2nd layer number of neurons.

# Create TF Model.
class NeuralNet(Model):
    # Set layers.
    def __init__(self):
        super(NeuralNet, self).__init__(name="NeuralNet")
        # First fully-connected hidden layer.
        self.fc1 = layers.Dense(64, activation=tf.nn.relu)
        # Second fully-connected hidden layer.
        self.fc2 = layers.Dense(128, activation=tf.nn.relu)
        # Third fully-connecter hidden layer.
        self.out = layers.Dense(num_classes, activation=tf.nn.softmax)

    # Set forward pass.
    def __call__(self, x, is_training=False):
        x = self.fc1(x)
        x = self.out(x)
        if not is_training:
            # tf cross entropy expect logits without softmax, so only
            # apply softmax when not training.
            x = tf.nn.softmax(x)
        return x
    
    def transform(self, x):
        a = tf.convert_to_tensor(x,dtype=float)
        a = tf.reshape(a,[1,30])
        return a

# Build neural network model.
neural_net = NeuralNet()

# Cross-Entropy loss function.
def cross_entropy(y_pred, y_true):
    # Encode label to a one hot vector.
    y_true = tf.cast(y_true, dtype=float)
    # Clip prediction values to avoid log(0) error.
    y_pred = tf.clip_by_value(y_pred, 1e-9, 1.)
    # Compute cross-entropy.
    return tf.reduce_mean(-tf.reduce_sum(tf.math.multiply(y_true , tf.math.log(y_pred))))

# Accuracy metric.
def accuracy(y_pred, y_true):
    # Predicted class is the index of highest score in prediction vector (i.e. argmax).
    correct_prediction = tf.equal(tf.argmax(y_pred, 1), tf.cast(y_true, tf.int64))
    return tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# Adam optimizer.
optimizer = tf.keras.optimizers.Adam(learning_rate)

# Optimization process. 
def run_optimization(x, y):
    # Wrap computation inside a GradientTape for automatic differentiation.
    with tf.GradientTape() as g:
        pred = neural_net(x, is_training=True)
        loss = cross_entropy(pred, y)

        # Compute gradients.
        gradients = g.gradient(loss, neural_net.trainable_variables)

        # Update W and b following gradients.
        optimizer.apply_gradients(zip(gradients, neural_net.trainable_variables))

# batch_x = []
# batch_y = []
# for i in range(0,len(data),31):
#         batch_x.append(data[i:i+30])
#         batch_y.append(data[i+30])

# train_data = tf.data.Dataset.from_tensor_slices((batch_x, batch_y))
# train_data = train_data.repeat().shuffle(5000).batch(batch_size).prefetch(1)

# # Run training for the given number of steps.
# for step, (batch_x, batch_y) in enumerate(train_data.take(training_steps), 1):
#     # Run the optimization to update W and b values.
#     run_optimization(batch_x, batch_y)
    
#     if step % display_step == 0:
#         pred = neural_net(batch_x)
#         loss = cross_entropy(pred, batch_y)
#         print("step: %i, loss: %f" % (step, loss))
#     if step % training_steps == 0:
#             plt.plot(range(200),tf.argmax(batch_y, 1))
#             plt.plot(range(200),tf.argmax(pred, 1))
#             plt.legend(labels=['human','model'])
#             plt.show()

# # Save TF model.
# neural_net.save_weights(filepath="./model/tfmodel.ckpt")


# # Re-build neural network model with default values.
# neural_net = NeuralNet()
# # Load saved weights.
# neural_net.load_weights(filepath="./model/tfmodel.ckpt")
# a = tf.convert_to_tensor(batch_x[0],dtype=float)
# a = tf.reshape(a,[1,30])
# pred = neural_net(a)
# print(pred)