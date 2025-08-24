import matplotlib
import matplotlib.pyplot as plt
import pathlib

from train import loadData
from prediction import loadThetas

def make_plot(X, Y, theta0=None, theta1=None, out="price_vs_km.png", show=False):
    plt.figure()
    plt.scatter(X, Y, s=18, alpha=0.85, label="data")

    if theta0 is not None and theta1 is not None:
        x0, x1 = min(X), max(X)
        y0 = theta0 + theta1 * x0
        y1 = theta0 + theta1 * x1
        plt.plot([x0, x1], [y0, y1], linewidth=2.0, label="linear fit")

    plt.xlabel("Mileage (km)")
    plt.ylabel("Price (€)")
    plt.title("Car price vs mileage")
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.legend(loc="best")
    plt.tight_layout()

    if show:
        # If you set Agg backend above, this won't open a window—remove the Agg line for GUI use.
        plt.show()
    else:
        pathlib.Path(out).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out, dpi=150)
        print(f"✔ Saved figure to {out}")

if __name__ == "__main__":
    X, Y = loadData()
    theta0, theta1 = loadThetas()
    make_plot(X, Y, theta0, theta1)