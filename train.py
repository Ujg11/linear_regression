import pandas as pd
import sys
import json
import pathlib

# We turn bad cells into NaN and then drop any NaN rows
# We remove negatives (no sense)
# We sort values
# Returns X (the data we want to analize) and Y (value we want to predict)
def loadData(dataFile="data.csv"):
	try:
		df = pd.read_csv(dataFile)
	except FileNotFoundError:
		sys.exit(f"Error: file not found: {dataFile}")

	requiredCol = {"km", "price"}
	for c in requiredCol:
		if c not in df.columns:
			sys.exit(f"Error: missing 'km' or 'price' column in data file named {dataFile}")

	df["km"] = pd.to_numeric(df["km"], errors="coerce")
	df["price"] = pd.to_numeric(df["price"], errors="coerce")
	df = df.dropna(subset=["km", "price"])

	df = df[(df["km"] >= 0) & (df["price"] >= 0)]

	df = df.sort_values("km").reset_index(drop=True)

	if df.empty:
		sys.exit("Error: no valid rows after cleaning data.csv")

	X = df["km"].astype(float).to_list()
	Y = df["price"].astype(float).to_list()
	return X, Y


# This function calculates the loss
# X and Y must have the same length
def lossFunctionMSE(X, Y, B, W):
    assert len(X) == len(Y)
    m = len(X)
    mse = 0.0
    for i in range(m):
        e = (B + W * X[i]) - Y[i] 
        mse += e * e
    return mse / m


# Updates one step of B and W at the same time
# Formula B: B1 = B0 - lr * 1/m * sum(Y^ - Y)
# Formula W: W1 = W0 - lr * 1/m * sum(Y^ - Y) * X
# This two formulas apeare when we derivate the MSE function
def gradientStepRaw(X, Y, B, W, lr_b, lr_w):
    assert len(X) == len(Y), "X and Y must have the same length"
    m = len(X)

    sum_e = 0.0
    sum_e_x = 0.0
    for i in range(m):
        e = (B + W * X[i]) - Y[i]
        sum_e += e
        sum_e_x += e * X[i]

    dB = sum_e / m
    dW = sum_e_x / m

    B = B - lr_b * dB
    W = W - lr_w * dW
    return B, W


# MODEL: ^Y = B + W * X = θ0 + θ1 * X
# Every logEvery iterations we print the information
def gradientDescent(X, Y, lr_b=1e-6, lr_w=1e-5, epochs=300_000, logEvery=100_000):
    """
    Learn B (bias) and W (slope) directly on raw mileage.
    """
    B = 0.0  # θ0
    W = 0.0  # θ1
    for t in range(epochs):
        B, W = gradientStepRaw(X, Y, B, W, lr_b, lr_w)
        if logEvery and (t % logEvery == 0):
            J = lossFunctionMSE(X, Y, B, W)
            print(f"epoch {t:>8} | cost {J:>12.4f} | B {B:>12.6f} | W {W:>12.9f}")
    return B, W


def saveThetas(theta0, theta1, path="thetas.json"):
    data = {"theta0": float(theta0), "theta1": float(theta1)}
    pathlib.Path(path).write_text(json.dumps(data, indent=2))
    print(f"\nSaved parameters to {path}")
    print(f"theta0 = {theta0:.6f}")
    print(f"theta1 = {theta1:.12f}")



if __name__ == "__main__":
	X, Y = loadData("data.csv")
	kX = [x / 1000.0 for x in X]
	B, Wk = gradientDescent(kX, Y, lr_b=1e-3, lr_w=1e-6, epochs=50_000, logEvery=5_000)
	theta0 = B
	theta1 = Wk / 1000.0
	saveThetas(theta0, theta1)