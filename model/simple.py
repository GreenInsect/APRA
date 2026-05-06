import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleNet(nn.Module):
    def __init__(self, name=None, created_time=None):
        super(SimpleNet, self).__init__()
        self.name=name
        self.gradient = None
        self.created_time = created_time


    def save_stats(self, epoch, loss, acc):
        self.stats['epoch'].append(epoch)
        self.stats['loss'].append(loss)
        self.stats['acc'].append(acc)

    def activations_hook(self, grad):
        self.gradient = grad

    def copy_params(self, state_dict, coefficient_transfer=100):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name in own_state:
                own_state[name].copy_(param.clone())


class SimpleMnist(SimpleNet):
    def __init__(self, name=None): 
        super(SimpleMnist, self).__init__(name)
        # self.conv1 = nn.Conv2d(1, 20, kernel_size=5)
        # self.conv2 = nn.Conv2d(20, 50, kernel_size=5)
        # self.conv2_drop = nn.Dropout2d()
        # self.fc1 = nn.Linear(4 * 4 * 50, 500)
        # self.fc2 = nn.Linear(500, 10)
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4 * 4 * 50, 500)
        self.fc2 = nn.Linear(500, 10)

    def features(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        return x

    # def forward(self, x):
    #     x = F.relu(F.max_pool2d(self.conv1(x), 2))
    #     x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
    #     x = x.view(-1, 320)
    #     x = F.relu(self.fc1(x))
    #     x = F.dropout(x, training=self.training)
    #     x = self.fc2(x)
    #     return F.log_softmax(x, dim=1)

    def forward(self, x, latent=False):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        if x.requires_grad:
            x.register_hook(self.activations_hook)
        x = x.view(-1, 4 * 4 * 50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        out = F.log_softmax(x, dim=1)
        if latent:
            return out, x
        else:
            return out

    def first_activations(self, x):
        x = F.relu(self.conv1(x))
        return x

    def final_activations(self, x, layer=None):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        return x


class SupSimpleMnist(SimpleNet):
    def __init__(self, name=None):
        super(SupSimpleMnist, self).__init__(name)
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4 * 4 * 50, 500)
        # self.fc2 = nn.Linear(500, 10)

    def features(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        return x

    def forward(self, x, latent=False):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        # if x.requires_grad:
        #     x.register_hook(self.activations_hook)
        x = x.view(-1, 4 * 4 * 50)
        x = F.relu(self.fc1(x))
        # x = self.fc2(x)
        out = torch.flatten(x, 1)
        out = F.normalize(out, dim=1)
        return out
        # if latent:
        #     return out, x
        # else:
        #     return out

    def first_activations(self, x):
        x = F.relu(self.conv1(x))
        return x

    def final_activations(self, x, layer=None):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        return x