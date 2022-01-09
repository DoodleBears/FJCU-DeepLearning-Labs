import torch

print(torch.cuda.is_available())
# True (O)
# False (X)

x = torch.empty(2)
print(x)
# tensor([5.2430e+33, 5.4511e-43])

x = torch.empty(2, 3)
print(x)
# tensor([[0., 0., 0.],
#         [0., 0., 0.]])

if torch.cuda.is_available():
    device = torch.device("cuda")
    x = torch.ones(5, device=device)
    y = torch.ones(5)
    y = y.to(device)
    z = x + y
    print(z)