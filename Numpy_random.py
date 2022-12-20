from numpy import random
import numpy as np

# Generate a random integer from 0 to 100
x_int = random.randint(100)
# Generate a 1-D array containing 5 random integers from 0 to 100:
array_1_D = random.randint(100, size=5)
# Generate a 2-D array containing 5 random integers from 0 to 100:
array_2_D = random.randint(100, size=(5, 2))

# returns a random float between 0 and 1
x_float = random.rand()
# Generate a 1-D array containing 5 random floats
array_float = random.rand(5)
# Generate a 2-D array containing 5 random floats
array_float_2_D = random.rand(3, 5)

# Generate Random Number From Array
random_choice = random.choice([3, 5, 7, 9])
# Generate Random New 2-D Array From Old 1-D Array
random_array = random.choice([3, 5, 7, 9], size=(3, 5))
# Generate a random 1-D array with probability (p=1)
x = random.choice([3, 5, 7, 9], p=[0.1, 0.3, 0.6, 0.0], size=100)

# Random Permutations
# The shuffle() method makes changes to the original array.
arr = np.array([1, 2, 3, 4, 5])
random.shuffle(arr)
# The permutation() method returns a re-arranged array (and leaves the original array un-changed).
arr = np.array([1, 2, 3, 4, 5])
print(random.permutation(arr))

'''
Normal Distribution
It has three parameters:
loc - (Mean) where the peak of the bell exists.
scale - (Standard Deviation) how flat the graph distribution should be.
size - The shape of the returned array.
'''
normal = random.normal(size=(2, 3))
normal_2 = random.normal(loc=1, scale=2, size=(2, 3))

'''
Binomial Distribution
It has three parameters:
n - number of trials.
p - probability of occurence of each trial (e.g. for toss of a coin 0.5 each).
size - The shape of the returned array.
'''
binomial = random.binomial(n=10, p=0.5, size=10)

'''
Poisson Distribution
It has two parameters:
lam - rate or known number of occurences e.g. 2 for above problem.
size - The shape of the returned array.
'''
poisson = random.poisson(lam=2, size=10)

'''
Uniform Distribution
It has three parameters:
a - lower bound - default 0 .0.
b - upper bound - default 1.0.
size - The shape of the returned array.
'''
uniform = random.uniform(size=(2, 3))

'''
Logistic Distribution
Used extensively in machine learning in logistic regression, neural networks etc.
It has three parameters:
loc - mean, where the peak is. Default 0.
scale - standard deviation, the flatness of distribution. Default 1.
size - The shape of the returned array.
'''
logistic = random.logistic(loc=1, scale=2, size=(2, 3))

'''
Multinomial Distribution
e.g. Blood type of a population, dice roll outcome.
It has three parameters:
n - number of possible outcomes (e.g. 6 for dice roll).
pvals - list of probabilties of outcomes (e.g. [1/6, 1/6, 1/6, 1/6, 1/6, 1/6] for dice roll).
size - The shape of the returned array.
'''
multinomial = random.multinomial(n=6, pvals=[1 / 6, 1 / 6, 1 / 6, 1 / 6, 1 / 6, 1 / 6])

'''
Exponential Distribution
Exponential distribution is used for describing time till next event e.g. failure/success etc.
It has two parameters:
scale - inverse of rate ( see lam in poisson distribution ) defaults to 1.0.
size - The shape of the returned array.
'''
exponential = random.exponential(scale=2, size=(2, 3))

'''
Chi Square Distribution
Chi Square distribution is used as a basis to verify the hypothesis.
It has two parameters:
df - (degree of freedom).
size - The shape of the returned array.
'''
chi_square = random.chisquare(df=2, size=(2, 3))

'''
Rayleigh Distribution
Rayleigh distribution is used in signal processing.
It has two parameters:
scale - (standard deviation) decides how flat the distribution will be default 1.0).
size - The shape of the returned array.
'''
rayleigh = random.rayleigh(scale=2, size=(2, 3))

'''
Pareto Distribution
A distribution following Pareto's law i.e. 80-20 distribution (20% factors cause 80% outcome).
It has two parameter:
a - shape parameter.
size - The shape of the returned array.
'''
pareto = random.pareto(a=2, size=(2, 3))

'''
Zipf Distribution
It has two parameters:
a - distribution parameter.
size - The shape of the returned array.
'''
zipf = random.zipf(a=2, size=(2, 3))
