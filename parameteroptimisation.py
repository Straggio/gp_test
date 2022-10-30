from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_probability as tfp
from check_shapes import check_shapes
from matplotlib.axes import Axes

import gpflow

class LinearModel(gpflow.Module):
    @check_shapes(
        "slope: [n_inputs, n_outputs]",
        "bias: [n_outputs]",
    )
    def __init__(
        self, slope: gpflow.base.TensorData, bias: gpflow.base.TensorData
    ) -> None:
        super().__init__()
        self.slope = gpflow.Parameter(slope)
        self.bias = gpflow.Parameter(bias)

    @check_shapes(
        "X: [n_rows, n_inputs]",
        "return: [n_rows, n_outputs]",
    )
    def predict(self, X: gpflow.base.TensorData) -> tf.Tensor:
        return X @ self.slope + self.bias[:, None]

model = LinearModel([[1.0], [2.0]], [0.5])
gpflow.utilities.print_summary(model)

#optimisation
X_train = np.array([[0, 0], [0, 2], [1, 0], [3, 2]])
Y_train = np.array([[1], [7], [3], [13]])

def loss() -> tf.Tensor:
    Y_predicted = model.predict(X_train)
    squared_error = (Y_predicted - Y_train) ** 2
    return tf.reduce_mean(squared_error)

opt = gpflow.optimizers.Scipy()
opt.minimize(loss, model.trainable_variables)
gpflow.utilities.print_summary(model)

co2_data = pd.read_csv(
    "https://gml.noaa.gov/webdata/ccgg/trends/co2/co2_mm_mlo.csv", comment="#"
)
Xco2 = co2_data["decimal date"].values[:, None]
Yco2 = co2_data["average"].values[:, None]

def plot_co2_model_prediction(
    ax: Axes, model: gpflow.models.GPModel, start: float, stop: float
) -> None:
    Xplot = np.linspace(start, stop, 200)[:, None]
    idx_plot = (start < Xco2) & (Xco2 < stop)

    y_mean, y_var = model.predict_y(Xplot, full_cov=False)
    y_lower = y_mean - 1.96 * np.sqrt(y_var)
    y_upper = y_mean + 1.96 * np.sqrt(y_var)

    ax.plot(Xco2[idx_plot], Yco2[idx_plot], "kx", mew=2)
    (mean_line,) = ax.plot(Xplot, y_mean, "-")
    color = mean_line.get_color()
    ax.plot(Xplot, y_lower, lw=0.1, color=color)
    ax.plot(Xplot, y_upper, lw=0.1, color=color)
    ax.fill_between(
        Xplot[:, 0], y_lower[:, 0], y_upper[:, 0], color=color, alpha=0.1
    )


opt_options = dict(maxiter=100)


def plot_co2_kernel(
    kernel: gpflow.kernels.Kernel,
    *,
    optimize: bool = False,
) -> None:
    model = gpflow.models.GPR(
        (Xco2, Yco2),
        kernel=kernel,
    )
    if optimize:
        opt = gpflow.optimizers.Scipy()
        opt.minimize(
            model.training_loss, model.trainable_variables, options=opt_options
        )
    gpflow.utilities.print_summary(model)

    _, (ax1, ax2) = plt.subplots(nrows=1, ncols=2)
    plot_co2_model_prediction(ax1, model, 1950, 2050)
    plot_co2_model_prediction(ax2, model, 2015, 2030)

plot_co2_kernel(gpflow.kernels.SquaredExponential(), optimize=True)
plt.savefig('figures/optimisationSQE.pdf')

plot_co2_kernel(
    gpflow.kernels.SquaredExponential()
    + gpflow.kernels.Periodic(gpflow.kernels.SquaredExponential()),
    optimize=True,
)
plt.savefig('figures/optimisationSQE2.pdf')

plot_co2_kernel(
    gpflow.kernels.SquaredExponential(variance=25000.0 , lengthscales=200.0)
    + gpflow.kernels.Periodic(gpflow.kernels.SquaredExponential(), period=1.0),
    optimize=True,
)
plt.savefig('figures/optimisationSQE3.pdf')



model = gpflow.models.GPR(
    (Xco2, Yco2),
    kernel=gpflow.kernels.SquaredExponential(
        variance=25000.0, lengthscales=200.0
    )
    + gpflow.kernels.Periodic(gpflow.kernels.SquaredExponential(), period=1.0),
)

# opt = tf.keras.optimizers.Adam()
#
# @tf.function
# def step():
#     opt.minimize(model.training_loss, model.trainable_variables)
tfp.optimizer.differential_evolution_minimize(model.training_loss, initial_position = model.trainable_variables)

maxiter = 100
for i in range(maxiter):
    step()
    if i % 100 == 0:
        print(i, model.training_loss().numpy())

kernel = model.kernel
plot_co2_kernel(kernel, optimize=False)
plt.savefig('figures/optimisationSQE_DEM.pdf')
