from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from matplotlib.axes import Axes
from matplotlib.cm import coolwarm

import gpflow

def plot_2d_kernel_samples(ax: Axes, kernel: gpflow.kernels.Kernel) -> None:
    n_grid = 30
    X = np.zeros((0, 2))
    Y = np.zeros((0, 1))
    model = gpflow.models.GPR((X, Y), kernel=deepcopy(kernel))

    Xplots = np.linspace(-0.6, 0.6, n_grid)
    Xplot1, Xplot2 = np.meshgrid(Xplots, Xplots)
    Xplot = np.stack([Xplot1, Xplot2], axis=-1)
    Xplot = Xplot.reshape([n_grid ** 2, 2])

    tf.random.set_seed(20220905)
    fs = model.predict_f_samples(Xplot, num_samples=1)
    fs = fs.numpy().reshape((n_grid, n_grid))
    ax.plot_surface(Xplot1, Xplot2, fs, cmap=coolwarm)
    ax.set_title("Example $f$")


def plot_2d_kernel_prediction(ax: Axes, kernel: gpflow.kernels.Kernel) -> None:
    n_grid = 30
    # hide: begin
    # fmt: off
    # hide: end
    X = np.array(
        [
            [-0.4, -0.5], [0.1, -0.3], [0.4, -0.4], [0.5, -0.5], [-0.5, 0.3],
            [0.0, 0.5], [0.4, 0.4], [0.5, 0.3],
        ]
    )
    Y = np.array([[0.8], [0.0], [0.5], [0.3], [1.0], [0.2], [0.7], [0.5]])
    # hide: begin
    # fmt: on
    # hide: end
    model = gpflow.models.GPR(
        (X, Y), kernel=deepcopy(kernel), noise_variance=1e-3
    )
    gpflow.set_trainable(model.likelihood, False)
    opt = gpflow.optimizers.Scipy()
    opt.minimize(model.training_loss, model.trainable_variables)

    Xplots = np.linspace(-0.6, 0.6, n_grid)
    Xplot1, Xplot2 = np.meshgrid(Xplots, Xplots)
    Xplot = np.stack([Xplot1, Xplot2], axis=-1)
    Xplot = Xplot.reshape([n_grid ** 2, 2])

    f_mean, _ = model.predict_f(Xplot, full_cov=False)
    f_mean = f_mean.numpy().reshape((n_grid, n_grid))
    ax.plot_surface(Xplot1, Xplot2, f_mean, cmap=coolwarm, alpha=0.7)
    ax.scatter(X[:, 0], X[:, 1], Y[:, 0], s=50, c="black")
    ax.set_title("Example data fit")


def plot_2d_kernel(kernel: gpflow.kernels.Kernel) -> None:
    _, (samples_ax, prediction_ax) = plt.subplots(
        nrows=1, ncols=2, subplot_kw={"projection": "3d"}
    )
    plot_2d_kernel_samples(samples_ax, kernel)
    plot_2d_kernel_prediction(prediction_ax, kernel)

plot_2d_kernel(gpflow.kernels.SquaredExponential())
plt.savefig('figures/2d_SQE.pdf')

plot_2d_kernel(gpflow.kernels.SquaredExponential(lengthscales=[0.1, 0.5]))
plt.savefig('figures/2d_SQEwithParameters.pdf')
