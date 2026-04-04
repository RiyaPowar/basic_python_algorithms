import matplotlib.pyplot as plt
from scipy import stats
x = [5,7,8,7,2,17,2,9,4,11,12,9,6]
y = [99,86,87,88,111,86,103,87,94,78,77,85,86]

slope, intercept, r, p, std_err = stats.linregress(x,y)
print(f"Relation coefficient: {r}")

def myfunc(x):
    return slope * x + intercept

mymodel = list(map(myfunc, x))


plt.scatter(x,y)
plt.plot(x, mymodel)
plt.show()


"""
Strength of relationship is |r|, where r is the relation coefficient. The closer |r| is to 1, the stronger the relationship. In this case, r is approximately -0.76, indicating a strong negative relationship between x and y.
"""