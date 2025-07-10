# ft_linear_regression – 42 School

## Objective
Predict a car’s selling price from its mileage (`data.csv`) with **simple linear regression**, writing every line of the algorithm yourself.  
External ML libraries that “do the job for you” (scikit-learn, `numpy.polyfit`, etc.) are **forbidden**; only basic Python (and plain NumPy for array maths, if allowed) is permitted.

## Theory

### Model
For one input feature (mileage **X**) the linear model is a straight line  

$$
\hat{Y} \;=\; B \;+\; W\,X
$$

where  

* **B** – bias
* **W** – weight

Both parameters are learned from the data.

### Loss (Function) — Mean Squared Error
Prediction error is measured with Mean Squared Error (MSE):

$$
\mathcal{L}(B,W)
  = \frac{1}{m}\sum_{i=1}^{m}\bigl(\hat{Y}_{i}-Y_{i}\bigr)^2
$$

* $m$ – number of training samples  

### Gradient  
Taking partial derivatives of $\mathcal{L}$ with respect to each parameter gives  

$$
\frac{\partial\mathcal{L}}{\partial B}
  \;=\;
  \frac{1}{m}\sum_{i=1}^{m}\bigl(\hat{Y}_{i}-Y_{i}\bigr),
\qquad
\frac{\partial\mathcal{L}}{\partial W}
  \;=\;
  \frac{1}{m}\sum_{i=1}^{m}\bigl(\hat{Y}_{i}-Y_{i}\bigr)\,X_{i}.
$$

### Optimisation — Gradient Descent
The generic update rule is  

$$
\theta^{(t+1)} \;=\; \theta^{(t)} \;-\; \alpha\,
        \frac{\partial\mathcal{L}}{\partial\theta},
$$

so plugging the expressions above yields the concrete updates you’ll implement:

$$
\begin{aligned}
B^{(t+1)} &= B^{(t)}
            \;-\;
            \alpha\,
            \frac{1}{m}
            \sum_{i=1}^{m}(\hat{Y}_{i}-Y_{i}) \\[6pt]
W^{(t+1)} &= W^{(t)}
            \;-\;
            \alpha\,
            \frac{1}{m}
            \sum_{i=1}^{m}(\hat{Y}_{i}-Y_{i})\,X_{i}
\end{aligned}
$$

* **$\alpha$** – learning-rate (step size).

  Too large → divergence
  
  too small → slow convergence.

---

With this foundation, `train.py` will learn **B** and **W**, and `prediction.py` will use them to estimate prices for new mileage values.
