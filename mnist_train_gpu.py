import torch
import torchvision
import time
from torch import nn
from torch.nn import functional as F
from torch import optim
from utils import plot_image, one_hot, plot_curve


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 512

# step1.load dataset

train_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('mnist_data', train=True, download=True,
                               transform=torchvision.transforms.Compose([
                                   torchvision.transforms.ToTensor(),
                                   torchvision.transforms.Normalize((0.1307,), (0.3081,))
                               ])),
    batch_size=batch_size, shuffle=True)

test_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('mnist_data', train=False, download=True,
                               transform=torchvision.transforms.Compose([
                                   torchvision.transforms.ToTensor(),
                                   torchvision.transforms.Normalize((0.1307,), (0.3081,))
                               ])),
    batch_size=batch_size, shuffle=True)

x, y = next(iter(train_loader))
print('x shape is', x.shape, 'y shape is', y.shape, x.min(), x.max())
plot_image(x, y, 'image sample')


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.model = nn.Sequential(

            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 10),

        )

    def forward(self, x):
        x = self.model(x)
        return x

        """
        # x: [b, 1, 28, 28]
        # h1 = relu(xw1+b1)
        x = F.relu(self.fc1(x))
        # h2 = relu(h1w2+b2)
        x = F.relu(self.fc2(x))
        # h3 = h2w3+b3
        x = self.fc3(x)
        return x

        """


net = Net()
net.to(device)
# loss_fn = nn.CrossEntropyLoss()
loss_fn = nn.MSELoss()
loss_fn.to(device)

optimizer = optim.SGD(net.parameters(), lr=1e-2, momentum=0.9)

train_loss = []
time_start = time.time()  # 计时
epoch = 4
for i in range(epoch):

    for batch_idx, (x, y) in enumerate(train_loader):

        x = x.to(device)
        y = y.to(device)
        # x:[b, 1, 28, 28],  y:[512]
        # [b, 1, 28, 28]-> [b, feature]
        x = x.view(x.size(0), 28*28)
        out = net(x)
        y_onehot = one_hot(y)
        y_onehot = y_onehot.to(device)
        loss = loss_fn(out, y_onehot)

        # clear grad
        optimizer.zero_grad()
        # update grad
        loss.backward()
        # w' = w- lr*grad
        optimizer.step()
        train_loss.append(loss.item())

        if batch_idx % 10 == 0:
            print(i, batch_idx, loss.item())

time_end = time.time()
print('train time is', time_end-time_start)
plot_curve(train_loss)
# get optimal [w1, b1, w2, b2, w3, b3]

total_correct = 0
for x, y in test_loader:

    x = x.to(device)
    y = y.to(device)

    x = x.view(x.size(0), 28*28)
    out = net(x)
    pred = out.argmax(dim=1)
    correct = pred.eq(y).sum().float().item()
    total_correct += correct

total_num = len(test_loader.dataset)
acc = total_correct / total_num
print('test accuracy', acc)


x, y = next(iter(test_loader))

x = x.to(device)
y = y.to(device)

out = net( )
pred = out.argmax(dim=1)
plot_image(x, pred, 'test')


