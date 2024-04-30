$$
(\frac{1}{1 - \mid (pred - true) \mid} + C \mathbf{1}_{\text{segno sbagliato}})^2 \text{ if | pred - true | < 1}\\
(\mid (pred - true) \mid + C \mathbf{1}_{\text{segno sbagliato}})^2 \text{altrimenti}
$$

what about a loss like this:
* the problem with MSE is that it downscales numbers in the range 0-1 because the square of such numbers is smaller than the original. Maybe it is better to penalize such numbers more heavily so that gradients are higher: that would be what the first case is for
* the second case is just MSE because error is greatear than 1
* the C is a penalization term almost like the hardness in SVMs: we add a penalty if the sign is misclassified. The choice of such penalty is a hyperparam. 
  * What do you think a good range for C would be? As $pred - true$ goes to 0, we have that y goes to 1, meaning that a good base value for C may be $1$ or close to it
  * an intuition for the choice of C is that we may want to first correct big regression errors and then focus on sign correction, so a smaller C would help with that, because C would become more relevant as $\frac{1}{1 - \mid (pred - true) \mid}$ shrinks
  * a huge C may mean that we just focus on sign correction

Revised loss function:
$$
(\frac{3}{3 - \mid (pred - true) \mid} + C \mathbf{1}_{\text{segno sbagliato}})^2 - 1
$$

- the 3 there is to be interpreted as $\text{error_range_upper_bound} + 1$, so that gradients do not explode
- the $-1$ at the end is so that the perfect prediction (error = 0 and correct sign) has a loss of 0. Without it we would have gradient updates even if the model does perfect