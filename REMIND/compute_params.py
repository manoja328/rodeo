import torch
from torchsummary import summary
from torchvision.models import resnet50

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = 'cpu'
model = resnet50().to(device)

# summary(model, [(3, 1000, 1333)], device=device)

for name, p in model.named_parameters():
    print(name, p.shape)
