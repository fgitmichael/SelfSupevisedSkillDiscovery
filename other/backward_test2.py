import torch
from itertools import chain

lin1 = torch.nn.Linear(1, 1)
lin2 = torch.nn.Linear(1, 1)
lin3 = torch.nn.Linear(1, 1)

opt1 = torch.optim.SGD(
    chain(
        lin1.parameters(),
        lin2.parameters()
    ),
    lr=0.1
)
opt2 = torch.optim.SGD(
    lin3.parameters(),
    lr=0.1
)

a = torch.tensor([2.])

b = lin1(a)
c = lin2(b)
d = lin3(c.detach())

opt1.zero_grad()
c.backward()
opt1.step()

opt2.zero_grad()
d.backward()
opt2.step()
