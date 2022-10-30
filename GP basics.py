# Imports
#%matplotlib inline
#%config InlineBackend.figure_format = 'svg'

import numpy as np
import scipy
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.gridspec as gridspec
import seaborn as sns


# Set matplotlib and seaborn plotting style
sns.set_style('darkgrid')
np.random.seed(35)
#

def covariance_matrix (x_a,x_b):
    # sigma = 1
    norm = -0.5* scipy.spatial.distance.cdist(x_a, x_b, 'sqeuclidean')
    return np.exp(norm)


#def covariance_matrix (x_a,x_b):


# =============================================================================
# # Illustrate covariance matrix and function
#
# # Show covariance matrix example from exponentiated quadratic
# fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 3))
# xlim = (-3, 3)
# X = np.expand_dims(np.linspace(*xlim, 25), 1)
# Σ = covariance_matrix(X, X)
# # Plot covariance matrix
# im = ax1.imshow(Σ, cmap=cm.YlGnBu)
# cbar = plt.colorbar(
#     im, ax=ax1, fraction=0.045, pad=0.05)
# cbar.ax.set_ylabel('$k(x,x)$', fontsize=10)
# ax1.set_title((
#     'Exponentiated quadratic \n'
#     'example of covariance matrix'))
# ax1.set_xlabel('x', fontsize=13)
# ax1.set_ylabel('x', fontsize=13)
# ticks = list(range(xlim[0], xlim[1]+1))
# ax1.set_xticks(np.linspace(0, len(X)-1, len(ticks)))
# ax1.set_yticks(np.linspace(0, len(X)-1, len(ticks)))
# ax1.set_xticklabels(ticks)
# ax1.set_yticklabels(ticks)
# ax1.grid(False)
#
# # Show covariance with X=0
# xlim = (-4, 4)
# X = np.expand_dims(np.linspace(*xlim, num=50), 1)
# zero = np.array([[0]])
# Σ0 = covariance_matrix(X, zero)
# # Make the plots
# ax2.plot(X[:,0], Σ0[:,0], label='$k(x,0)$')
# ax2.set_xlabel('x', fontsize=13)
# ax2.set_ylabel('covariance', fontsize=13)
# ax2.set_title((
#     'Exponentiated quadratic  covariance\n'
#     'between $x$ and $0$'))
# # ax2.set_ylim([0, 1.1])
# ax2.set_xlim(*xlim)
# ax2.legend(loc=1)
#
# fig.tight_layout()
# plt.show()
# =============================================================================

#sampling from prior
nb_of_samples = 40
number_of_functions = 5
#independent variable samples
X = np.expand_dims(np.linspace(-4, 4, nb_of_samples), 1)
sigma = covariance_matrix(X, X)
#draw samples from the prior at our data points
#mean = 0
Y = np.random.multivariate_normal(mean=np.zeros(nb_of_samples), cov=sigma, size=number_of_functions)
#plot the functions
# =============================================================================
# plt.figure(figsize=(6,4))
# for i in range(number_of_functions):
#     plt.plot(X, Y[i], linestyle = '-', marker = 'o', markersize = 2)
# plt.xlabel('x')
# plt.ylabel('y = f(x)')
# plt.xlim([-4,4])
# plt.show()
# =============================================================================

def GP (X1, y1, X2, kernel_func):
    #Kernel of Observations
    sigma11 = kernel_func(X1, X1)
    #Kernel of observation vs prediction
    sigma12 = kernel_func(X1, X2)
    #Solve with scipy
    res = scipy.linalg.solve(sigma11, sigma12, assume_a='pos').T
    #Compute pos_mean
    mean2 = res @ y1
    #Compute pos covariance
    sigma22 = kernel_func(X2, X2)
    sigma2 = sigma22 - (res @ sigma12)
    return mean2, sigma2

def GP_withnoise (X1, y1, X2, kernel_func, noise):
    #Kernel of Observations
    sigma11 = kernel_func(X1, X1) + ((noise **2)* np.eye(n1))
    #Kernel of observation vs prediction
    sigma12 = kernel_func(X1, X2)
    #Solve with scipy
    res = scipy.linalg.solve(sigma11, sigma12, assume_a='pos').T
    #Compute pos_mean
    mean2 = res @ y1
    #Compute pos covariance
    sigma22 = kernel_func(X2, X2)
    sigma2 = sigma22 - (res @ sigma12)
    return mean2, sigma2





#define function
f_sinus = lambda x: (np.sin(x)*3).flatten()
#f_sinus = lambda x: (np.exp(x)*0.2).flatten()
#f_sinus = lambda x: (np.arctan(x)*5).flatten()

n1 = 15
n2 = 75
ny = 5
domain = (-8,8)

#sample observations
#X1 = np.random.uniform(domain[0]-2, domain[1]-2, size = (n1,1))
X1 = np.random.exponential(5, size = (n1,1))-6
y1 = f_sinus(X1)

X2 = np.linspace(domain[0], domain[1], n2).reshape(-1,1)

print(X1)

mean2, sigma2 = GP(X1, y1, X2, covariance_matrix)

variance2 = np.sqrt(np.diag(sigma2))

y2 = np.random.multivariate_normal(mean=mean2, cov=sigma2, size =ny)

# Plot the postior distribution and some samples
fig, (ax1, ax2) = plt.subplots(
    nrows=2, ncols=1, figsize=(6, 6))
# Plot the distribution of the function (mean, covariance)
ax1.plot(X2, f_sinus(X2), 'b--', label='$sin(x)$')
ax1.fill_between(X2.flat, mean2-2*variance2, mean2+2*variance2, color='red',
                 alpha=0.15, label='$2 \sigma_{2|1}$')
ax1.plot(X2, mean2, 'r-', lw=2, label='$\mu_{2|1}$')
ax1.plot(X1, y1, 'ko', linewidth=2, label='$(x_1, y_1)$')
ax1.set_xlabel('$x$', fontsize=13)
ax1.set_ylabel('$y$', fontsize=13)
ax1.set_title('Distribution of posterior and prior data.')
ax1.axis([domain[0], domain[1], -8, 8])
ax1.legend()
# Plot some samples from this function
ax2.plot(X2, y2.T, '-')
ax2.set_xlabel('$x$', fontsize=13)
ax2.set_ylabel('$y$', fontsize=13)
ax2.set_title('5 different function realizations from posterior')
ax1.axis([domain[0], domain[1], -8, 8])
ax2.set_xlim([-8, 8])
plt.tight_layout()
plt.show()
plt.savefig('figures/withoutnoise1.pdf', bbox_inches='tight')

noise = 0.5
y1 = y1 + ((noise ** 2) * np.random.randn(n1))

mean2, sigma2 = GP_withnoise(X1, y1, X2, covariance_matrix, noise)

variance2 = np.sqrt(np.diag(sigma2))

y2 = np.random.multivariate_normal(mean=mean2, cov=sigma2, size =ny)

# Plot the postior distribution and some samples
fig, (ax1, ax2) = plt.subplots(
    nrows=2, ncols=1, figsize=(6, 6))
# Plot the distribution of the function (mean, covariance)
ax1.plot(X2, f_sinus(X2), 'b--', label='$sin(x)$')
ax1.fill_between(X2.flat, mean2-2*variance2, mean2+2*variance2, color='red',
                 alpha=0.15, label='$2 \sigma_{2|1}$')
ax1.plot(X2, mean2, 'r-', lw=2, label='$\mu_{2|1}$')
ax1.plot(X1, y1, 'ko', linewidth=2, label='$(x_1, y_1)$')
ax1.set_xlabel('$x$', fontsize=13)
ax1.set_ylabel('$y$', fontsize=13)
ax1.set_title('Distribution of posterior and prior data.')
ax1.axis([domain[0], domain[1], -8, 8])
ax1.legend()
# Plot some samples from this function
ax2.plot(X2, y2.T, '-')
ax2.set_xlabel('$x$', fontsize=13)
ax2.set_ylabel('$y$', fontsize=13)
ax2.set_title('5 different function realizations from posterior')
ax1.axis([domain[0], domain[1], -8, 8])
ax2.set_xlim([-8, 8])
plt.tight_layout()
plt.show()
plt.savefig('figures/withnoise1.pdf', bbox_inches='tight')
