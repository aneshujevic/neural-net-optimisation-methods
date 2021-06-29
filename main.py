import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist

# Hyperparameters
LEARNING_RATE = 0.001

# Adam
BETA_1 = 0.9
BETA_2 = 0.999
EPSILON = 1e-08

# Momentum
MOMENTUM_BETA = 0.9

# RMSprop
RMS_BETA = 0.999
EPSILON_RMSPROP = 1e-07

# Neural network
NUMBER_NODES = 300

# Optimisation variables
EPOCHS = 10
BATCH_SIZE = 128

# NN variables
W1 = None
W2 = None
b1 = None
b2 = None
x_test = None
x_train = None
y_test = None
y_train = None


def get_batch(x_data, y_data, batch_size):
    indexes = np.random.randint(0, len(y_data), batch_size)
    return x_data[indexes, :, :], y_data[indexes]


def train_step(x_input, W1, b1, W2, b2):
    # flatten the input image from 28 x 28 to 784
    x_input = tf.reshape(x_input, (x_input.shape[0], -1))
    x = tf.add(tf.matmul(tf.cast(x_input, tf.float32), W1), b1)
    x = tf.nn.sigmoid(x)
    logits = tf.add(tf.matmul(x, W2), b2)
    return logits


def loss_fn(logits, labels):
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits))
    return cross_entropy


def train_model(x_train, y_train, x_test, y_test, batch_size, epochs, W1, b1, W2, b2, optimizer):
    # number of batches in total
    total_batch = int(len(y_train) / batch_size)
    results = []

    for epoch in range(epochs):
        avg_loss = 0
        for i in range(total_batch):
            batch_x, batch_y = get_batch(x_train, y_train, batch_size=batch_size)
            # create tensors
            batch_x = tf.Variable(batch_x)
            batch_y = tf.Variable(batch_y)
            # create a one hot vector
            batch_y = tf.one_hot(batch_y, 10)
            with tf.GradientTape() as tape:
                logits = train_step(batch_x, W1, b1, W2, b2)
                loss = loss_fn(logits, batch_y)
            gradients = tape.gradient(loss, [W1, b1, W2, b2])
            optimizer.apply_gradients(zip(gradients, [W1, b1, W2, b2]))
            avg_loss += loss / total_batch
        test_logits = train_step(x_test, W1, b1, W2, b2)
        max_idxs = tf.argmax(test_logits, axis=1)
        test_acc = np.sum(max_idxs.numpy() == y_test) / len(y_test)
        print(f"epoch: {epoch + 1}, loss={avg_loss:.3f}, test set accuracy={test_acc * 100:.3f}%")
        results.append((epoch, avg_loss.numpy(), test_acc * 100))

    return results


def plot_results(nn_results, min_y, max_y, x_label, y_label, transform_fun, title):
    fig, ax = plt.subplots()
    ax.plot(list(map(lambda x: int(x[0]) + 1, nn_results)), list(map(transform_fun, nn_results)))
    ax.set(xlabel=x_label, ylabel=y_label, title=title)
    ax.axis([1, EPOCHS, min_y, max_y])
    fig.savefig(f"{title.split('/')[0]}.png")
    plt.show()


def get_loss(result_tuple):
    return result_tuple[1]


def get_accuracy(result_tuple):
    return result_tuple[2]


def init_setup():
    global x_train, x_test, y_train, y_test, W1, W2, b1, b2
    # Load 60_000 rows of 28x28 pixel grayscale images for training
    # and 10_000 rows of same dimension images for testing
    # x_train: (60,000 x 28 x 28), y_train: (60,000)
    # x_test: (10,000 x 28 x 28), y_test: (10,000)
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    # normalize the input images i.e. scale them to [0, 1] segment
    x_train = x_train / 255.0
    x_test = x_test / 255.0
    # convert x_test to tensor so it can pass through model
    # train data will be converted on the fly
    x_test = tf.Variable(x_test)
    # now declare the weights connecting the input to the hidden layer
    # flattened images represent 784 input nodes (28 x 28)
    # with NUMBER_NODES hidden layer nodes
    W1 = tf.Variable(tf.random.normal([784, NUMBER_NODES], stddev=0.03), name='W1')
    b1 = tf.Variable(tf.random.normal([NUMBER_NODES]), name='b1')
    # and the weights connecting the hidden layer to the output layer
    # 10 output nodes (10 numbers :) )
    W2 = tf.Variable(tf.random.normal([NUMBER_NODES, 10], stddev=0.03), name='W2')
    b2 = tf.Variable(tf.random.normal([10]), name='b2')


def main():
    # setup the optimizer
    optimizer_none = tf.keras.optimizers.SGD(LEARNING_RATE, 0)
    optimizer_momentum = tf.keras.optimizers.SGD(LEARNING_RATE, MOMENTUM_BETA)
    optimizer_RMSProp = tf.keras.optimizers.RMSprop(LEARNING_RATE, RMS_BETA, epsilon=EPSILON_RMSPROP)

    # train the model
    print("Gradient descent without optimizer")
    init_setup()
    results_grad_descent = train_model(x_train, y_train, x_test, y_test, BATCH_SIZE, EPOCHS, W1, b1, W2, b2,
                                       optimizer_none)
    print()

    print("Gradient descent with momentum")
    init_setup()
    results_momentum = train_model(x_train, y_train, x_test, y_test, BATCH_SIZE, EPOCHS, W1, b1, W2, b2,
                                   optimizer_momentum)
    print()

    print("Gradient descent with RMSProp")
    init_setup()
    results_RMSprop = train_model(x_train, y_train, x_test, y_test, BATCH_SIZE, EPOCHS, W1, b1, W2, b2,
                                  optimizer_RMSProp)
    print()

    optimizer_adam = tf.keras.optimizers.Adam(LEARNING_RATE, BETA_1, BETA_2, EPSILON)

    print("Gradient descent with Adam")
    init_setup()
    results_adam = train_model(x_train, y_train, x_test, y_test, BATCH_SIZE, EPOCHS, W1, b1, W2, b2, optimizer_adam)
    print()

    print("Summary:")
    plot_results(results_grad_descent, 0, 3, "Epochs", "Average Loss", get_loss, title="No optimization loss/epoch")
    plot_results(results_grad_descent, 0, 100, "Epochs", "Test set accuracy", get_accuracy,
                 title="No optimization accuracy/epoch")

    plot_results(results_momentum, 0, 3, "Epochs", "Average Loss", get_loss, title="Momentum optimization loss/epoch")
    plot_results(results_momentum, 0, 100, "Epochs", "Test set accuracy", get_accuracy,
                 title="Momentum optimization accuracy/epoch")

    plot_results(results_RMSprop, 0, 3, "Epochs", "Average Loss", get_loss, title="RMSProp optimization loss/epoch")
    plot_results(results_RMSprop, 0, 100, "Epochs", "Test set accuracy", get_accuracy,
                 title="RMSprop optimization accuracy/epoch")

    plot_results(results_adam, 0, 3, "Epochs", "Average Loss", get_loss, title="Adam optimization loss/epoch")
    plot_results(results_adam, 0, 100, "Epochs", "Test set accuracy", get_accuracy,
                 title="Adam optimization accuracy/epoch")


if __name__ == "__main__":
    main()
