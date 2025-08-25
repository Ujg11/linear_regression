import json
import pathlib
import sys

def loadThetas(path="thetas.json"):
	p = pathlib.Path(path)
	if not p.exists():
		print("No training yet, execute 'train.py' first. Now we'll use θ0 = θ1 = 0.0.")
		return 0.0, 0.0
	try:
		data = json.loads(p.read_text())
		return float(data["theta0"]), float(data["theta1"])
	except Exception as e:
		print(f"Error: {e}.\n\nWe'll use θ0 = θ1 = 0.0")
		return 0.0, 0.0

def getMileage():
	while True:
		mileage_raw = input("Enter mileage in Km: ").strip()
		if not mileage_raw:
			print("Error: empty input.")
			continue
		mileage_raw = mileage_raw.replace(",", ".")
		try:
			mileage = float(mileage_raw)
		except ValueError:
			print("Error: please enter a numeric value")
			continue
		if mileage < 0:
			print("Error: mileage cannot be negative")
			continue
		return mileage

def estimatePrice(mileage, t0, t1):
	return t0 + mileage * t1

if __name__ == '__main__':
	mileage = getMileage()
	theta0, theta1 = loadThetas()
	price = estimatePrice(mileage, theta0, theta1)
	print(f"The estimated price is {price:.0f}€")