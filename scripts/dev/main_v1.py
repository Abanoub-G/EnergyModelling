import os
import sys
sys.path.append('../')

import random 
import matplotlib.pyplot as plt
import numpy as np
import datetime
import copy

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF

# from utils.common import set_random_seeds, set_cuda, logs

# =====================================================
# == Declarations
# =====================================================
SEED_NUMBER              = 0
USE_CUDA                 = True


DATASET_DIR              = '' 
DATASET_NAME             = "" # Options: 


MODEL_CHOICE             = "" # Option:
MODEL_VARIANT            = "" # Common Options: 

MODEL_DIR                = "../models/" + MODEL_CHOICE
MODEL_SELECTION_FLAG     = 2 # create an untrained model = 0, start from a pytorch trained model = 1, start from a previously saved local model = 2

# MODEL_FILENAME     = MODEL_VARIANT +"_"+DATASET_NAME+".pt"
# MODEL_FILEPATH     = os.path.join(MODEL_DIR, MODEL_FILENAME)


def main():

	# ========================================
	# == Preliminaries
	# ========================================
	# Fix seeds to allow for repeatable results 
	random_seed = 0
	np.random.seed(random_seed)
	random.seed(random_seed)

	# ========================================
	# == Setup Dataset 
	# ========================================
	# Generate data
	X = np.linspace(start=0, stop=10, num=1_000).reshape(-1, 1)
	y = np.squeeze(X * np.sin(X))

	# Plot data
	plt.plot(X, y, label=r"$f(x) = x \sin(x)$", linestyle="dotted")
	plt.legend()
	plt.xlabel("$x$")
	plt.ylabel("$f(x)$")
	_ = plt.title("True generative process")
	plt.savefig("dataset.png")

	# Select random points from dataset to create training data
	rng = np.random.RandomState(1)
	training_indices = rng.choice(np.arange(y.size), size=6, replace=False)
	X_train, y_train = X[training_indices], y[training_indices]

	# ========================================
	# == Training (non noisy)
	# ========================================
	# Now, we fit a Gaussian process on these few training 
	# data samples. We will use a radial basis function (RBF) 
	# kernel and a constant parameter to fit the amplitude.

	kernel = 1 * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2))
	gaussian_process = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=9)
	gaussian_process.fit(X_train, y_train)
	print(gaussian_process.kernel_)

	# After fitting our model, we see that the hyperparameters 
	# of the kernel have been optimized. Now, we will use our kernel
	# to compute the mean prediction of the full dataset and plot 
	# the 95% confidence interval.
	mean_prediction, std_prediction = gaussian_process.predict(X, return_std=True)
	plt.clf()
	plt.plot(X, y, label=r"$f(x) = x \sin(x)$", linestyle="dotted")
	plt.scatter(X_train, y_train, label="Observations")
	plt.plot(X, mean_prediction, label="Mean prediction")
	plt.fill_between(
		X.ravel(),
		mean_prediction - 1.96 * std_prediction,
		mean_prediction + 1.96 * std_prediction,
		alpha=0.5,
		label=r"95% confidence interval",
	)
	plt.legend()
	plt.xlabel("$x$")
	plt.ylabel("$f(x)$")
	_ = plt.title("Gaussian process regression on noise-free dataset")
	plt.savefig("Fitted_Gaussian_model.png")


	# ========================================
	# == Training (noisy)
	# ========================================
	# We add some random Gaussian noise to the 
	# target with an arbitrary standard deviation.
	noise_std = 0.75
	y_train_noisy = y_train + rng.normal(loc=0.0, scale=noise_std, size=y_train.shape)

	# We create a similar Gaussian process model. In addition to the kernel, 
	# this time, we specify the parameter alpha which can be interpreted as 
	# the variance of a Gaussian noise.
	gaussian_process = GaussianProcessRegressor(
		kernel=kernel, alpha=noise_std**4, n_restarts_optimizer=9
	)
	gaussian_process.fit(X_train, y_train_noisy)
	mean_prediction, std_prediction = gaussian_process.predict(X, return_std=True)

	# Let’s plot the mean prediction and the uncertainty region as before.
	plt.clf()
	plt.plot(X, y, label=r"$f(x) = x \sin(x)$", linestyle="dotted")
	plt.errorbar(
		X_train,
		y_train_noisy,
		noise_std,
		linestyle="None",
		color="tab:blue",
		marker=".",
		markersize=10,
		label="Observations",
	)
	plt.plot(X, mean_prediction, label="Mean prediction")
	plt.fill_between(
		X.ravel(),
		mean_prediction - 1.96 * std_prediction,
		mean_prediction + 1.96 * std_prediction,
		color="tab:orange",
		alpha=0.5,
		label=r"95% confidence interval",
	)
	plt.legend()
	plt.xlabel("$x$")
	plt.ylabel("$f(x)$")
	_ = plt.title("Gaussian process regression on a noisy dataset")
	plt.savefig("Fitted_Gaussian_model_noisy.png")







if __name__ == "__main__":

	main()