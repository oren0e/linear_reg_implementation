import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Minimize the function (x + 5)^2
cur_x = 3
rate = 0.01
precision = 0.000001
previous_step_size = 1
max_iters = 10000
iters = 0
def dfunc(x):
    return 2*(x+5)

while (previous_step_size > precision) & (iters < max_iters):
    prev_x = cur_x # store current x value in previous x
    cur_x = cur_x - rate * dfunc(prev_x)  # gradient descent
    previous_step_size = abs(cur_x - prev_x)  # change in x
    iters += 1
    print("Iteration ",iters,"\nX value is ",cur_x)

print("The local minimum occurs at ", cur_x," and is equal to ",dfunc(cur_x))

# 2. Minimize the function y = b0 + b1*X (linear regression)
X = 2 * np.random.rand(100,1)
y = 4 +3 * X+np.random.randn(100,1)

# plot the variables
plt.scatter(X,y)
plt.xlabel('X')
plt.ylabel('y')
plt.show()

def loss(X, beta, y):
    n = len(y)
    h = np.dot(X, beta)
    return (1/2*n) * np.sum(np.square(h - y))

def gradient(X, beta, y):
    n = len(y)
    h = np.dot(X, beta)
    return 1/n * np.sum((X.T).dot(h - y))

# def estimate(X, y, rate=0.001, precision=0.000001, max_iters=10000):



cur_beta = np.random.randn(2,1)
X_intercept = np.c_[np.ones((len(X),1)), X]
rate = 0.001
precision = 0.000001
previous_step_size = 1
max_iters = 10000
iters = 0

# for graphing
beta_array = np.zeros((max_iters,2))
loss_array = np.zeros(max_iters)

while (previous_step_size > precision) & (iters < max_iters):
    prev_beta = cur_beta
    cur_beta = cur_beta - rate * gradient(X_intercept, cur_beta, y)
    previous_step_size = abs(cur_beta[1] - prev_beta[1])
    iters += 1

    # store, for graphing
    beta_array[iters,:] = cur_beta.T
    loss_array[iters] = loss(X_intercept, cur_beta, y)

    print("Iteration ", iters, "\nBeta value is ", cur_beta)

print("The final MSE ", gradient(X_intercept, cur_beta, y)," and beta is ",cur_beta)

# loss plot
fig, ax = plt.subplots()
ax.set_ylabel('MSE')
ax.set_xlabel('Iterations')
ax.plot(range(1,iters), loss_array[1:iters], 'b.')
plt.show()

# plot the fits
fig, ax = plt.subplots(sharey=True)
ax.plot(X, y, 'b.')
ax.set_xlabel("$x_1$")
ax.set_ylabel("$y$")
for i in range(1,iters):
    ax.plot(X, np.dot(X_intercept, beta_array[i]), 'r-', alpha=0.1)
    #plt.show()
plt.show()

# TODO: 1. write a function for estimating
#       2. write 2 functions for plotting the loss and the fit with params: learning_rate
#       3. write a class named CustomLinearRegression which will have public methods fit(), plot_loss() and plot_fit()
#          and private methods loss(), gradient().
#       4. The fit method should also print to the screen more nicely the beta_hat and the final MSE.






plt.plot(X, np.dot(X_intercept, cur_beta), 'r-')
plt.plot(X, y, 'b.')
plt.xlabel("$x_1$")
plt.ylabel("$y$")
plt.show()
