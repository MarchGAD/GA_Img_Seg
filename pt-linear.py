import torch
import torch.optim as opt
import torch.nn as nn
from .util import calculate_diff
import matplotlib.pyplot as plt
class LinearSeg(nn.Module):

    def __init__(self, input_features) -> None:
        super().__init__()
        self.model = nn.Sequential(
            nn.Flatten(), 
            nn.Linear(input_features, 300, bias=True), 
            nn.ReLu(), 
            nn.Linear(300, 1, bias=True)
        )

    def forward(self, raw_input):
        thresh = self.model(raw_input)
        return thresh

    def loss(self, raw_input, thresh):
        return calculate_diff(raw_input, thresh)


def train(input_img, iters, feature_dim= 140 * 205):
    model = LinearSeg(feature_dim)
    sgd = opt.SGD(model.parameters, lr=0.001)
    for i in range(iters):
        i = i + 1
        thresh = model(input_img)
        loss = calculate_diff(input_img, thresh)
        sgd.zero_grad()
        loss.backward()
        sgd.step()
        print('step {}, loss is {%.4f}'.format(i, loss.detach().cpu()))

# if __name__ == '__main__':
    