import os
import sys
sys.path.append('../')

import random 
import matplotlib.pyplot as plt
import numpy as np
import datetime
import copy
import pandas as pd

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


# from utils.common import set_random_seeds, set_cuda, logs

# =====================================================
# == Declarations
# =====================================================
SEED_NUMBER              = 0
USE_CUDA                 = True


DATASET_DIR              = '../datasets/GR712RC_LEON3_POWER_MODEL_DATA/' 
DATASET_NAME             = "data/LEON3_BEEBS_finegrain.data" # Options: 


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
	# Load data
	file_path = DATASET_DIR+DATASET_NAME
	try:
		df = pd.read_csv(file_path, sep='\t', index_col=False)
	except FileNotFoundError:
		print(f"File not found: {file_path}")
	except Exception as e:
		print(f"An error occurred: {e}")

	print(df)  

	# Defind PMU counters names
	pmu_names = ["icmiss", "ichold", "dcmiss", "dchold", "wbhold", "ainst", "iinst", "bpmiss", "ahbutil", "ahbtutil", "branch", "call", "type2", "ldst", "load", "store"]

	# Extract benchmarks
	benchmarks = df["Benchmark"].unique()
	print(benchmarks)

	# ==================================================================
	# Step 1: Organize the dataset
	for benchmark in benchmarks:
		result_df = df[df['Benchmark'] == benchmark]
		print(result_df)
		plt.clf()

		runs = result_df["Run(#)"].unique()
		print(runs)
		# for run in [1]:#runs: TODO: In the futrue to take each point
		# 	result_df2 = result_df[result_df['Run(#)'] == run]
		# 	print(result_df2)

		# Extract dataset
		y = result_df[["Power[W]"]].values
		X = result_df[pmu_names].values 

		print((X))
		print(len(y))
	# ==================================================================
	# Step 2: Organize the dataset
		# Step 2: Split the dataset into training and testing sets
		X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

	# ==================================================================
	# Step 3: Train a Gaussian Process Regressor
		kernel = ConstantKernel() * RBF()
		gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, random_state=42)
		gp.fit(X_train, y_train)

	# ==================================================================
	# Step 4: Test the trained model
		y_pred = gp.predict(X_test)

		# Evaluate the performance using mean squared error
		mse = mean_squared_error(y_test, y_pred)
		print(f'Mean Squared Error on Test Set: {mse}')
		# print(len(y_test))
		# print(len(y_pred))

	# ==================================================================
	# Step 5: Plot actual vs predicted values
		plt.plot(y_test, label='Actual', marker='o')
		plt.plot(y_pred, label='Predicted', marker='x')
		plt.title('Actual vs Predicted Power Consumption')
		plt.xlabel('Data Points')
		plt.ylabel('Power Consumption')
		plt.legend()
		plt.ylim(0,4)
		plt.savefig("Results for "+benchmark+".png")
		# plt.scatter(y_test, y_pred, color='blue', label='Actual vs Predicted')
		# plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], linestyle='--', color='red', linewidth=2, label='Ideal Line')
		# plt.title('Actual vs Predicted Power Consumption')
		# plt.xlabel('Actual Power')
		# plt.ylabel('Predicted Power')
		# plt.legend()
		# plt.show()
		# plt.savefig("Results for "+benchmark+".pdf")

		# input("wait")
	# ===================================================
		





if __name__ == "__main__":

	main()