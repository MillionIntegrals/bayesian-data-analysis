import torch
import torch.distributions as td
import torch.optim as opt
import numpy as np


TPrior = td.Normal(0.0, 1.0)

torch.manual_seed(0)
data = torch.randn(100).to(torch.float64) + 3.0

# Parameters
mu = torch.tensor(0.0, dtype=torch.float64, requires_grad=True)
logstd = torch.tensor(0.0, dtype=torch.float64, requires_grad=True)


def get_elbo():
    q_distribution = td.Normal(mu, logstd.exp())
    theta_sample = q_distribution.rsample()

    log_prob_theta_model = TPrior.log_prob(theta_sample)
    log_prob_data_model = td.Normal(theta_sample, 1).log_prob(data).sum()

    # Second bit, variational distribution log prob, assuming theta is the hidden param
    log_prob_q = q_distribution.log_prob(theta_sample)

    loss = log_prob_theta_model + log_prob_data_model - log_prob_q

    # import torch.autograd

    # print("mu_grad=", torch.autograd.grad(loss, [mu], retain_graph=True))

    return loss


NUM_STEPS = 1001
optimizer = opt.SGD([mu, logstd], lr=0.001, momentum=0.1)


for step in range(NUM_STEPS):
    optimizer.zero_grad()
    loss = -get_elbo()
    loss.backward()
    optimizer.step()

    if step % 100 == 0:
        print("step {} loss = {}".format(step, loss))

print("MU=", mu.item())
print("LOGSTD=", logstd.item())

# For this simple (conjugate) model we know the exact posterior. In
# particular we know that the variational distribution should be
# centered near 3.0. So let's check this explicitly.
assert abs(mu.item() - 3.0) < 0.1


