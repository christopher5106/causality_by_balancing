import math
import torch
from torch import optim
import torch.nn.functional as F
import numpy as np

N=10

# data A -> B
true_p_a = np.random.random(N)
true_p_a /= np.sum(true_p_a)
true_p_b_a = np.random.random((N, N))
true_p_b_a /= np.sum(true_p_b_a, axis=1, keepdims=True)

# model parameters
p_a = torch.randn(N) / math.sqrt(N)
p_a.requires_grad_()

p_b_a = torch.randn(N, N) / N
p_b_a.requires_grad_()

p_b = torch.randn(N) / math.sqrt(N)
p_b.requires_grad_()

p_a_b = torch.randn(N, N) / N
p_a_b.requires_grad_()
#
# + parameter de structure gamma


optimizer = optim.Adam([p_a, p_b_a, p_b, p_a_b], lr=0.001)
b_size = 30
n_iters = 100000
running_loss = 0.0
for i in range(n_iters):
    a = np.random.choice(N, b_size, p=true_p_a, replace=True)
    b = np.array([np.random.choice(N, 1, p=true_p_b_a[_a])[0] for _a in a])

    optimizer.zero_grad()

    loss_a = - torch.mean(F.log_softmax(p_a, dim=0)[a])
    loss_b = - torch.mean(F.log_softmax(p_b_a, dim=1)[a, b])
    loss = 0.25 * loss_a + 0.25 * loss_b

    loss_b = - torch.mean(F.log_softmax(p_b, dim=0)[b])
    loss_a = - torch.mean(F.log_softmax(p_a_b, dim=1)[b, a])
    loss += 0.25 * loss_a + 0.25 * loss_b

    loss.backward()
    optimizer.step()

    running_loss += loss.item()
    if i % 2000 == 1999:    # print every 2000 mini-batches
        print('[%5d] loss: %.3f' % (i+1, running_loss/2000))
        running_loss = 0.0

print('Finished Training')
print(f"True p_a {true_p_a} {torch.nn.functional.softmax(p_a, dim=0)}")
print(f"True p_b_a {true_p_b_a} {torch.nn.functional.softmax(p_b_a, dim=1)}")


# eval for different balancing values
print("\nEval by cross entropy")
n_weight_sampling = 10
for _ in range(n_weight_sampling):
    running_loss_a_b, running_loss_b_a = 0.0, 0.0
    for i in range(200):
        a = np.random.choice(N, b_size, p=true_p_a, replace=True)
        b = np.array([np.random.choice(N, 1, p=true_p_b_a[_a])[0] for _a in a])

        running_loss_a_b -= torch.mean(F.log_softmax(p_a, dim=0)[a]
                                    + F.log_softmax(p_b_a, dim=1)[a, b])
        running_loss_b_a -= torch.mean(F.log_softmax(p_b, dim=0)[b]
                                    + F.log_softmax(p_a_b, dim=1)[b, a])

    print("Model A->B", running_loss_a_b.item())
    print("Model B->A", running_loss_b_a.item())
    print("Best model", "B->A" if running_loss_b_a < running_loss_a_b else "A->B")


n_weight_sampling = 10
print("\nEval by balanced cross entropy")
for _ in range(n_weight_sampling):
    true_p_a = np.random.random(N)
    true_p_a /= np.sum(true_p_a)
    weights_a = [1]*N # torch.rand(N)
    # weights_a /= torch.sum(weights_a)
    #
    # weights_b = np.random.random(N)
    # weights_b /= np.sum(weights_b)

    running_loss_a_b, running_loss_b_a = 0.0, 0.0
    for i in range(200):
        a = np.random.choice(N, b_size, p=true_p_a, replace=True)
        b = np.array([np.random.choice(N, 1, p=true_p_b_a[_a])[0] for _a in a])

        running_loss_a_b -= torch.mean(weights_a[a] * (F.log_softmax(p_a, dim=0)[a]
            + F.log_softmax(p_b_a, dim=1)[a, b]))
        running_loss_b_a -= torch.mean(weights_a[a] * (F.log_softmax(p_b, dim=0)[b]
            + F.log_softmax(p_a_b, dim=1)[b, a]))

    print("Model A->B", running_loss_a_b.item())
    print("Model B->A", running_loss_b_a.item())
    print("Best model", "B->A" if running_loss_b_a < running_loss_a_b else "A->B")

# you can do the same with weights_b

# TODO train the structure
