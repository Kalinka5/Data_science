import seaborn as sns
from numpy import random
import matplotlib.pyplot as plt

# Visualization of Normal Distribution
sns.displot(random.normal(size=1000), kind="kde")
plt.savefig('distribution/Normal_Distribution.png')

# Visualization of Binomial Distribution
sns.displot(random.binomial(n=10, p=0.5, size=1000), kind="kde")
plt.savefig('distribution/Binomial_Distribution.png')

# Visualization of Poisson Distribution
sns.displot(random.poisson(lam=2, size=1000))
plt.savefig('distribution/Poisson_Distribution.png')

# Visualization of Uniform Distribution
sns.displot(random.uniform(size=1000), kind="kde")
plt.savefig('distribution/Uniform_Distribution.png')

# Visualization of Logistic Distribution
sns.displot(random.logistic(size=1000), kind="kde")
plt.savefig('distribution/Logistic_Distribution.png')

# Visualization of Multinomial Distribution
sns.displot(random.multinomial(n=6, pvals=[1/6, 1/6, 1/6, 1/6, 1/6, 1/6], size=1000), kind="kde")
plt.savefig('distribution/Multinomial_Distribution.png')

# Visualization of Exponential Distribution
sns.displot(random.exponential(size=1000), kind="kde")
plt.savefig('distribution/Exponential_Distribution.png')

# Visualization of Chi Square Distribution
sns.displot(random.chisquare(df=1, size=1000), kind="kde")
plt.savefig('distribution/Chi_Square_Distribution.png')

# Visualization of Rayleigh Distribution
sns.displot(random.rayleigh(size=1000), kind="kde")
plt.savefig('distribution/Rayleigh_Distribution.png')

# Visualization of Pareto Distribution
sns.displot(random.pareto(a=2, size=1000), kind="kde")
plt.savefig('distribution/Pareto_Distribution.png')

# Visualization of Zipf Distribution
zipf = random.zipf(a=2, size=1000)
sns.displot(zipf[zipf < 10], kind="kde")
plt.savefig('distribution/Zipf_Distribution.png')
