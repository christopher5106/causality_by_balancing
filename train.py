import math
import torch
import numpy as np

N=10
n_epochs=10

# data A -> B
n_samples=100000
true_p_a = np.random.random(N)
true_p_a /= np.sum(true_p_a)
true_p_b_a = np.random.random((N, N))
true_p_b_a /= np.sum(true_p_b_a, axis=1, keepdims=True)

# model parameters
p_a = torch.randn(N) / math.sqrt(N)
p_a.requires_grad_()

p_b_a = torch.randn(N, N) / N
p_b_a.requires_grad_()

# p_b = torch.randn(N) / math.sqrt(N)
# p_b.requires_grad_()
#
# p_a_b = torch.randn(N, N) / N
# p_a_b.requires_grad_()
#
# + parameter de structure gamma

lr = 0.0001
for epoch in range(n_epochs):
    print(f"Epoch {epoch}")

    running_loss = 0.0
    for i in range(n_samples):
        # print(f"iter {i}")
        a = np.random.choice(N, 1, p=true_p_a)[0]
        b = np.random.choice(N, 1, p=true_p_b_a[a])[0]

        # # zero the parameter gradients
        # optimizer.zero_grad()

        # forward + backward + optimize
        loss = - torch.log(p_a[a]) #- torch.log(p_b_a[a, b])

        # outputs = net(inputs)
        # loss = criterion(outputs, labels)
        # loss.backward()
        # optimizer.step()

        loss.backward()
        with torch.no_grad():
            p_a -= p_a.grad * lr
            # p_b_a -= p_b_a.grad * lr
            p_a.grad.zero_()
            # p_b_a.grad.zero_()

        # # print statistics
        # running_loss += loss.item()
        # if i % 2000 == 1999:    # print every 2000 mini-batches
        #     print('[%d, %5d] loss: %.3f' %
        #           (epoch + 1, i + 1, running_loss / 2000))
        #     running_loss = 0.0
    print(f"True p_a {true_p_a} {p_a}")
    # print(f"True p_b_a {true_p_b_a} {p_b_a}")

print('Finished Training')
print(f"True p_a {true_p_a} {p_a}")
print(f"True p_b_a {true_p_b_a} {p_b_a}")
