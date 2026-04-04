import numpy as np
import matplotlib.pyplot as plt

class SimpleRegression:

    def __init__(self):
        self.coefficients_ = None
        self.intercept_ = None
        self.r2score_ = None

    def fit(self, X, y):
        n = len(X)
        X_b = np.c_[np.ones((n,1)), X] #Add x0 = 1 to each instance. This is done to account for the intercept term in the linear regression model. By adding a column of ones to the input data X, we can represent the intercept as a coefficient that multiplies this column of ones. This allows us to compute both the slope and the intercept in a single matrix operation when we calculate the coefficients using the normal equation.
        print(f"Design matrix X_b: {X_b}")
        self.coefficients_ = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y) #
        print(f"Coefficients: {self.coefficients_}")
        self.intercept_ = self.coefficients_[0]
        print(f"Intercept: {self.intercept_}")
        y_pred =  X_b.dot(self.coefficients_)
        print(f"Predicted values: {y_pred}")

        #R^2
        self.r2score_ = 1 - (np.sum((y - y_pred) ** 2) / np.sum((y - np.mean(y))**2))
        print(f"R^2 score: {self.r2score_}")
        self.y_pred_ = y_pred

    def predict(self, X):
        X_b = np.c_[np.ones((len(X),1)), X]
        return X_b.dot(self.coefficients_)



X_simple = np.array([1,2,3,4,5,6,7,8,9,10]).reshape(-1,1)
#It is required to reshape the data to make it a 2D array, as the fit method expects a 2D array for X. The reshape(-1,1) converts the 1D array into a 2D array with one column and as many rows as needed.
#[1,2,3,4,5,6,7,8,9,10] is converted to [[1],[2],[3],[4],[5],[6],[7],[8],[9],[10]].
y_simple = np.array([2,4,5,4,5,7,8,9,10,12])

slr = SimpleRegression()
slr.fit(X_simple, y_simple)

print(f"Simple LR Coefficients: {slr.coefficients_}")
print(f"R² Score: {slr.r2score_:.2f}")

plt.scatter(X_simple, y_simple, color='blue', label='Data Points')
plt.plot(X_simple, slr.y_pred_, color='red', label='Regression Line')
plt.title('Simple Linear Regression')
plt.xlabel("X")
plt.ylabel("y")
plt.legend()
plt.show()


"""
self.coefficients_ = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y) is basically θ=(XbT​Xb​)−1XbT​y
The main goal of linear regression is to find the best fitting values of coefficients which are slope and intercept so that the r squared error is minimized. The formula θ=(XbT​Xb​)−1XbT​y is derived from the normal equation of linear regression, which provides a closed-form solution to find the optimal coefficients that minimize the cost function (mean squared error) for linear regression.
Linear regression converts the scattered data points into a straight line that best fits the data.
"""
