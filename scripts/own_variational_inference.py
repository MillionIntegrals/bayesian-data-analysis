import torch
import torch.distributions as td
import torch.optim as opt
import numpy as np
import matplotlib.pyplot as plt


from bda.distributions import *


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


TX = torch.from_numpy(X).to(torch.float64)

mu = torch.tensor(2.0, dtype=torch.float64, requires_grad=True)
std = torch.tensor(4.0, dtype=torch.float64, requires_grad=True)

TPrior = td.Normal(0, 1)


def calculate_elbo_for_theta(theta_sample):
    # Our variational distribution
    q_distribution = td.Normal(mu, std)
    # All samples are calculated with respect to q distribution, therefore we sample from q
    # theta_sample = q_distribution.sample()

    # First bit, model log prob assuming theta is the hidden param
    log_prob_theta_model = TPrior.log_prob(theta_sample)
    print("lptm=", log_prob_theta_model)
    log_prob_data_model = td.Normal(theta_sample, 1).log_prob(TX).sum()
    print("lpdm=", log_prob_data_model)

    # Second bit, variational distribution log prob, assuming theta is the hidden param
    log_prob_q = q_distribution.log_prob(theta_sample)

    print("lpq=", log_prob_q)

    return log_prob_theta_model + log_prob_data_model - log_prob_q


optimizer = opt.SGD([mu, std], lr=0.001, momentum=0.1)
optimizer.zero_grad()

loss = -calculate_elbo_for_theta(2.592709575848736)
print("loss =", loss.item())
loss.backward()

optimizer.step()

# print("ELBO=", calculate_elbo_for_theta(2.985611735833615))
# print("STOP")
