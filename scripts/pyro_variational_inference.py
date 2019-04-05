import pyro
import pyro.infer
import pyro.optim
import torch
import numpy as np
import scipy.stats as stats
import torch.distributions.constraints as constraints
import matplotlib.pyplot as plt


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


N = 100

Prior = Normal(0, 1)


hidden_theta = -0.30876849393399736
X = np.array([ 0.83882339, -0.18327733, -2.1367785 , -0.23884984,  0.64975018,
        0.3822017 , -1.28156452, -1.76602789, -0.52034258, -1.02277046,
       -0.1081116 , -0.04086724, -1.00239476, -2.31659967, -2.01284933,
       -0.04255648, -0.48477407, -0.85343536,  0.57087556, -1.67664997,
       -0.16917258,  0.33454645, -1.7441229 ,  1.04228389,  0.33730209,
        0.85081516,  0.16614794, -0.87870244,  0.22860209, -1.12077229,
        0.00912435, -1.34767148, -1.06479637,  0.51134094,  1.65901749,
       -1.00470063,  0.88676986,  0.0328493 , -0.53138992, -0.26180192,
       -0.18376393, -1.06245679,  1.33091922, -1.14943535, -0.03175903,
       -0.29296988,  0.76961376, -0.80308002, -0.59702679, -0.73219416,
        0.36688673, -1.35749804, -0.69708232,  1.08306657, -2.09771641,
       -1.50261039,  0.07427721,  0.25997902, -2.71479907,  1.45067433,
       -0.24867818, -0.25115161,  1.00637968,  0.48491776, -1.87099424,
       -0.29619929,  0.43546577,  0.24719993, -0.14766969, -1.22047931,
       -0.6958676 ,  0.50736201, -0.02641034, -0.00787796, -2.67367672,
        2.22168505, -0.45168883,  0.20791519, -0.94687149, -0.41716543,
       -1.66260982,  1.17850728, -0.04458167,  0.21673739, -0.77691316,
       -0.1035381 , -1.60944244,  1.28643517,  0.7185452 , -2.15315254,
        0.39248977, -1.88790874, -0.70113703, -0.07157591, -0.67460936,
       -1.10845035,  0.46250522,  0.45822605,  0.02116005, -0.58546019])


def model(size):
    theta = pyro.sample("continuous_prior", pyro.distributions.Normal(0, 1))

    data = pyro.sample(
        "continuous_likelihood",
        pyro.distributions.Normal(theta, 1).expand_by([size]),
        obs=torch.from_numpy(X).to(torch.float64)
    )

    print("theta=", theta.item())
    # print("data=", data)

    return data


pyro.param("c_a", torch.tensor(2.0).to(torch.float64))
pyro.param("c_b", torch.tensor(4.0).to(torch.float64), constraint=constraints.positive)


def guide(size):
    mu = pyro.param("c_a", torch.tensor(1.0))
    std = pyro.param("c_b", torch.tensor(1.0), constraint=constraints.positive)
    # print("mu=", mu)
    # print("Std=", std)
    guide_sample = pyro.sample("continuous_prior", pyro.distributions.Normal(mu, std))

    # print("guide_sample=", guide_sample)

    return guide_sample


continuous_svi = pyro.infer.SVI(
    model=model,
    guide=guide,
    optim=pyro.optim.SGD({"lr": 0.001, "momentum": 0.1}),
    loss=pyro.infer.Trace_ELBO()
)

# continuous_svi.evaluate_loss(size=X.size)
continuous_svi.step(X.size)

print("STOP")