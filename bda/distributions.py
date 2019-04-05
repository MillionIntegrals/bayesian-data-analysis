import scipy.stats as stats


# In the meantime I'll define thin wrappers around the probability distributions
class Bernoulli:
    def __init__(self, p):
        self.p = p

    def sample(self, size=1):
        return stats.bernoulli.rvs(p=self.p, size=size)


class Uniform:
    def __init__(self, start, end):
        self.start = start
        self.end = end

    def sample(self, size=1):
        return stats.uniform.rvs(loc=self.start, scale=self.end - self.start, size=size)

    def pdf(self, x):
        return stats.uniform.pdf(x, loc=self.start, scale=self.end - self.start)

    def mean(self):
        return stats.uniform.mean(loc=self.start, scale=self.end - self.start)


class Beta:
    def __init__(self, alpha, beta):
        self.alpha = alpha
        self.beta = beta

    def pdf(self, X):
        return stats.beta.pdf(X, a=self.alpha, b=self.beta)

    def mean(self):
        return stats.beta.mean(a=self.alpha, b=self.beta)


class Normal:
    def __init__(self, mu, sigma):
        self.mu = mu
        self.sigma = sigma

    def pdf(self, X):
        return stats.norm.pdf(X, loc=self.mu, scale=self.sigma)

    def sample(self, size=1):
        return stats.norm.rvs(loc=self.mu, scale=self.sigma, size=size)

    def mean(self):
        return self.mu

