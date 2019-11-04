def linear_pred(x, w0, w1):
    return w1 * x + w0


def plot_data(dataset, line_weights=None):
    import matplotlib.pyplot as plt

    x_axis_min, x_axis_max = 4, 10

    x_vals = [x for x, _ in dataset]
    y_vals = [y for _, y in dataset]
    plt.plot(x_vals, y_vals, 'ro')

    if line_weights is not None:
        ya = linear_pred(x_axis_min, line_weights[0], line_weights[1])
        yb = linear_pred(x_axis_max, line_weights[0], line_weights[1])
        plt.plot([x_axis_min, x_axis_max], [ya, yb])

    plt.axis([x_axis_min, x_axis_max, 20, 40])
    plt.show()
