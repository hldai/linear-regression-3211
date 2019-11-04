import random
from utils import plot_data, linear_pred


def loss(dataset, w0, w1):
    l = 0
    for x, y in dataset:
        y_pred = linear_pred(x, w0, w1)
        l += (y_pred - y) ** 2
    return l


def train_with_gradient_descent(dataset):
    learning_rate = 0.01
    n_iter = 5000
    n_samples = len(dataset)
    # w0, w1 = random.uniform(-1, 1), random.uniform(-1, 1)
    w0, w1 = 0, 0

    loss_val = loss(dataset, w0, w1)
    print('initial loss={}'.format(loss_val))

    for t in range(n_iter):
        grad_w0, grad_w1 = 0, 0
        for x, y in dataset:
            y_pred = linear_pred(x, w0, w1)
            grad_w0 += y_pred - y
            grad_w1 += x * (y_pred - y)
        grad_w0 *= 2 / n_samples
        grad_w1 *= 2 / n_samples
        w0 = w0 - learning_rate * grad_w0
        w1 = w1 - learning_rate * grad_w1

        loss_val = loss(dataset, w0, w1)
        if (t + 1) % 500 == 0:
            print('iter {}, loss={}'.format(t + 1, loss_val))

    plot_data(dataset, (w0, w1))


def train_with_stochastic_gradient_descent(dataset):
    learning_rate = 0.01
    n_iter = 5000

    # w0, w1 = random.uniform(-1, 1), random.uniform(-1, 1)
    w0, w1 = 0, 0

    loss_val = loss(dataset, w0, w1)
    print('initial loss={}'.format(loss_val))

    for t in range(n_iter):
        for x, y in dataset:
            y_pred = linear_pred(x, w0, w1)
            grad_w0 = y_pred - y
            grad_w1 = x * (y_pred - y)
            w0 = w0 - learning_rate * grad_w0
            w1 = w1 - learning_rate * grad_w1

        loss_val = loss(dataset, w0, w1)
        if (t + 1) % 500 == 0:
            print('iter {}, loss={}'.format(t + 1, loss_val))

    plot_data(dataset, (w0, w1))


if __name__ == '__main__':
    random.seed(3211)

    dataset = [(6.2, 26.3), (6.5, 26.65), (5.48, 25.03), (6.54, 26.01), (7.18, 27.9),
               (7.93, 30.47)]

    # __plot_data(dataset)

    # train_with_gradient_descent(dataset)
    train_with_stochastic_gradient_descent(dataset)
