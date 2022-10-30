from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from matplotlib.axes import Axes
from matplotlib.cm import coolwarm

import gpflow

#def functions to plot kernels with samples and predictions

def plot_kernel_samples(ax: Axes, kernel: gpflow.kernels.Kernel) -> None:
    X = np.zeros((0, 1))
    Y = np.zeros((0, 1))
    model = gpflow.models.GPR((X, Y), kernel=deepcopy(kernel))
    Xplot = np.linspace(-0.6, 0.6, 100)[:, None]
    tf.random.set_seed(20220905)
    n_samples = 5
    # predict_f_samples draws n_samples examples of the function f, and returns their values at Xplot.
    fs = model.predict_f_samples(Xplot, n_samples)
    ax.plot(Xplot, fs[:, :, 0].numpy().T, label=kernel.__class__.__name__)
    ax.set_ylim(bottom=-2.0, top=2.0)
    ax.set_title("Example $f$s")


def plot_kernel_prediction(
    ax: Axes, kernel: gpflow.kernels.Kernel, *, optimise: bool = True
) -> None:
    X = np.array([[-0.5], [0.0], [0.4], [0.5]])
    Y = np.array([[1.0], [0.0], [0.6], [0.4]])
    model = gpflow.models.GPR(
        (X, Y), kernel=deepcopy(kernel), noise_variance=1e-3
    )

    if optimise:
        gpflow.set_trainable(model.likelihood, False)
        opt = gpflow.optimizers.Scipy()
        opt.minimize(model.training_loss, model.trainable_variables)

    Xplot = np.linspace(-0.6, 0.6, 100)[:, None]

    f_mean, f_var = model.predict_f(Xplot, full_cov=False)
    f_lower = f_mean - 1.96 * np.sqrt(f_var)
    f_upper = f_mean + 1.96 * np.sqrt(f_var)

    ax.scatter(X, Y, color="black")
    (mean_line,) = ax.plot(Xplot, f_mean, "-", label=kernel.__class__.__name__)
    color = mean_line.get_color()
    ax.plot(Xplot, f_lower, lw=0.1, color=color)
    ax.plot(Xplot, f_upper, lw=0.1, color=color)
    ax.fill_between(
        Xplot[:, 0], f_lower[:, 0], f_upper[:, 0], color=color, alpha=0.1
    )
    ax.set_ylim(bottom=-1.0, top=2.0)
    ax.set_title("Example data fit")


def plot_kernel(
    kernel: gpflow.kernels.Kernel, *, optimise: bool = True
) -> None:
    _, (samples_ax, prediction_ax) = plt.subplots(nrows=1, ncols=2)
    plot_kernel_samples(samples_ax, kernel)
    plot_kernel_prediction(prediction_ax, kernel, optimise=optimise)

#plot examples
plot_kernel(gpflow.kernels.Matern12())
plt.savefig('figures/kernelmatern12.pdf')

plot_kernel(gpflow.kernels.Matern32())
plt.savefig('figures/kernelmatern32.pdf')

plot_kernel(gpflow.kernels.Matern52())
plt.savefig('figures/kernelmatern52.pdf')

plot_kernel(gpflow.kernels.SquaredExponential())
plt.savefig('figures/kernelSquaredExponential.pdf')

plot_kernel(gpflow.kernels.ArcCosine())
plt.savefig('figures/kernelarccosine.pdf')

plot_kernel(gpflow.kernels.Cosine())
plt.savefig('figures/kernelcosine.pdf')

plot_kernel(gpflow.kernels.Polynomial(degree=2.0))
plt.savefig('figures/kernelpolynomial2.pdf')

plot_kernel(gpflow.kernels.Polynomial(degree=3.0))
plt.savefig('figures/kernelpolynomial3.pdf')

plot_kernel(gpflow.kernels.Polynomial(degree=4.0))
plt.savefig('figures/kernelpolynomial4.pdf')

plot_kernel(gpflow.kernels.Polynomial(degree=5.0))
plt.savefig('figures/kernelpolynomial5.pdf')

plt.close()

#compare kernels
_, ax = plt.subplots(nrows=1, ncols=1)
plot_kernel_prediction(ax, gpflow.kernels.Matern12())
plot_kernel_prediction(ax, gpflow.kernels.Matern32())
plot_kernel_prediction(ax, gpflow.kernels.Matern52())
plot_kernel_prediction(ax, gpflow.kernels.SquaredExponential())
ax.legend()
plt.savefig('figures/kernelcomparison.pdf')

_, bx = plt.subplots(nrows=1, ncols=1)
plot_kernel_prediction(bx, gpflow.kernels.Polynomial(degree=2.0))
plot_kernel_prediction(bx, gpflow.kernels.Polynomial(degree=3.0))
plot_kernel_prediction(bx, gpflow.kernels.Polynomial(degree=4.0))
plot_kernel_prediction(bx, gpflow.kernels.Polynomial(degree=5.0))
plt.savefig('figures/kernelpolynomialcomparison.pdf')

plot_kernel(gpflow.kernels.SquaredExponential(variance=0.1, lengthscales=0.2),optimise=False,)
plt.savefig('figures/parameter1.pdf')

plot_kernel(gpflow.kernels.SquaredExponential(variance=1, lengthscales=0.2),optimise=False,)
plt.savefig('figures/parameter2.pdf')

plot_kernel(gpflow.kernels.SquaredExponential(variance=1, lengthscales=1),optimise=False,)
plt.savefig('figures/parameter3.pdf')

plot_kernel(gpflow.kernels.SquaredExponential(variance=0.1, lengthscales=1),optimise=False,)
plt.savefig('figures/parameter4.pdf')

#switching parameters
_, bx =  plt.subplots(nrows =1, ncols=1)
plot_kernel_prediction(bx, gpflow.kernels.SquaredExponential(variance=0.1, lengthscales=0.2),optimise=False,)
plot_kernel_prediction(bx, gpflow.kernels.SquaredExponential(variance=1, lengthscales=0.2),optimise=False,)
plot_kernel_prediction(bx, gpflow.kernels.SquaredExponential(variance=1, lengthscales=1),optimise=False,)
plot_kernel_prediction(bx, gpflow.kernels.SquaredExponential(variance=0.1, lengthscales=1),optimise=False,)
plt.savefig('figures/parametercomparison.pdf')

plt.close()

#combining kernel
#periodic

plot_kernel(gpflow.kernels.Periodic(gpflow.kernels.SquaredExponential(), period=0.3))
plt.savefig('figures/SQE_periodic.pdf')

plot_kernel(gpflow.kernels.Periodic(gpflow.kernels.SquaredExponential(), period=0.5))
plt.savefig('figures/SQE_periodic2.pdf')

plot_kernel(gpflow.kernels.Periodic(gpflow.kernels.SquaredExponential(), period=0.1))
plt.savefig('figures/SQE_periodic3.pdf')

#change points
plot_kernel(gpflow.kernels.ChangePoints(kernels=[gpflow.kernels.Matern52(lengthscales=1.0),
                                                 gpflow.kernels.Matern12(lengthscales=3.0),
                                                 gpflow.kernels.SquaredExponential(lengthscales=2.0),],
                                                 locations=[-0.2, 0.2],steepness=50.0,),optimise=False,)

plt.savefig('figures/changepointsSQEvariants.pdf')

#addition
plot_kernel(gpflow.kernels.Periodic(gpflow.kernels.SquaredExponential()) + gpflow.kernels.Cosine())
plt.savefig('figures/additionSQE_Cosine.pdf')

plot_kernel(gpflow.kernels.Periodic(gpflow.kernels.SquaredExponential()) + gpflow.kernels.ArcCosine())
plt.savefig('figures/additionSQE_ArcCosine.pdf')

plot_kernel(gpflow.kernels.Periodic(gpflow.kernels.SquaredExponential()) + gpflow.kernels.Polynomial(degree=2.0))
plt.savefig('figures/additionSQE_polynomial.pdf')

plot_kernel(gpflow.kernels.Periodic(gpflow.kernels.SquaredExponential(), period=0.3) + gpflow.kernels.Polynomial(degree=2.0))
plt.savefig('figures/additionSQE_polynomial2.pdf')

plot_kernel(gpflow.kernels.Periodic(gpflow.kernels.SquaredExponential()) + gpflow.kernels.Polynomial(degree=3.0))
plt.savefig('figures/additionSQE_polynomial3.pdf')

#multiplication
plot_kernel(gpflow.kernels.Periodic(gpflow.kernels.SquaredExponential()) * gpflow.kernels.Cosine())
plt.savefig('figures/multiplicationSQE_Cosine.pdf')

plot_kernel(gpflow.kernels.Periodic(gpflow.kernels.SquaredExponential()) * gpflow.kernels.ArcCosine())
plt.savefig('figures/multiplicationSQE_ArcCosine.pdf')

plot_kernel(gpflow.kernels.Periodic(gpflow.kernels.SquaredExponential()) * gpflow.kernels.Polynomial(degree=2.0))
plt.savefig('figures/multiplicationSQE_polynomial.pdf')

plot_kernel(gpflow.kernels.Periodic(gpflow.kernels.SquaredExponential(), period=0.3) * gpflow.kernels.Polynomial(degree=2.0))
plt.savefig('figures/multiplicationSQE_polynomial2.pdf')

plot_kernel(gpflow.kernels.Periodic(gpflow.kernels.SquaredExponential()) * gpflow.kernels.Polynomial(degree=3.0))
plt.savefig('figures/multiplicationSQE_polynomial3.pdf')
