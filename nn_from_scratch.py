import numpy as np
from dataclasses import dataclass


@dataclass
class BinaryCrossEntropyLossPartials:
    output_wrt_yhat: np.ndarray


class BinaryCrossEntropyLoss:

    def forward(self, yhat: np.ndarray, y: np.ndarray):
        assert yhat.shape == y.shape, f"Shape difference: {yhat} vs {y}"
        assert np.all(1 >= yhat) and np.all(yhat >= 0), f"Domain error for yhat: {yhat}"
        assert np.all(1 >= y) and np.all(y >= 0), f"Domain error for y: {y}"
        # We drop the 1/len(yhat) factor to make the loss the same as torch's BCE loss
        loss = -(y * np.log(yhat) + (1 - y) * np.log(1 - yhat))
        # clamp the loss entries to at most 100 to avoid nan (and like torch's BCE loss)
        loss = np.clip(loss, -100, 100)
        return loss

    def backwards(
        self, yhat: np.ndarray, y: np.array
    ) -> BinaryCrossEntropyLossPartials:
        output_wrt_yhat = (yhat - y) / (yhat - yhat**2)

        return BinaryCrossEntropyLossPartials(output_wrt_yhat)


@dataclass
class LinearPartials:
    output_wrt_weight: np.ndarray
    output_wrt_bias: np.ndarray


class Linear:

    def __init__(self, in_features: int, out_features: int) -> None:
        self.in_features = in_features
        self.out_features = out_features
        self.weight = np.random.randn(out_features, in_features).astype(np.float32)
        self.bias = np.random.randn(out_features).astype(np.float32)

    def forward(self, x: np.ndarray) -> np.ndarray:
        return self.weight @ x + self.bias

    def backward(self, x: np.ndarray) -> LinearPartials:
        output_wrt_weight = x
        output_wrt_bias = np.ones_like(self.bias)
        return LinearPartials(output_wrt_weight, output_wrt_bias)


@dataclass
class SigmoidPartials:
    output_wrt_x: np.ndarray


class Sigmoid:

    def forward(self, x: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-x))

    def backward(self, x: np.ndarray) -> SigmoidPartials:
        output_wrt_x = np.exp(-x) / ((np.exp(-x) + 1) ** 2)
        return SigmoidPartials(output_wrt_x)


class MyNetwork:

    def __init__(self, in_features: int, out_features: int):
        self.in_features = in_features
        self.out_features = out_features

        self.linear1 = Linear(in_features, out_features)
        self.sigmoid = Sigmoid()
        self.loss = BinaryCrossEntropyLoss()

    def forward(self, x: np.ndarray):
        forward_out = self.linear1.forward(x)
        sigmoid_out = self.sigmoid.forward(forward_out)
        return sigmoid_out

    def forward_loss(self, x: np.ndarray, y: np.ndarray):
        sigmoid_out = self.forward(x)
        loss_out = self.loss.forward(sigmoid_out, y)
        return loss_out.sum()

    def update_weights(self, x: np.ndarray, y: np.ndarray, lr: float):
        forward_out = self.linear1.forward(x)
        sigmoid_out = self.sigmoid.forward(forward_out)

        loss_wrt_sigmoid = self.loss.backwards(sigmoid_out, y).output_wrt_yhat

        sigmoid_wrt_linear1 = self.sigmoid.backward(forward_out).output_wrt_x
        linear1_wrt_weight = self.linear1.backward(x).output_wrt_weight
        linear1_wrt_bias = self.linear1.backward(x).output_wrt_bias

        loss_wrt_linear1 = loss_wrt_sigmoid * sigmoid_wrt_linear1
        # Note: outer product because matrix multiplication!
        loss_wrt_weight = np.outer(loss_wrt_linear1, linear1_wrt_weight)
        loss_wrt_bias = loss_wrt_linear1 * linear1_wrt_bias

        self.linear1.weight = self.linear1.weight - lr * loss_wrt_weight
        self.linear1.bias = self.linear1.bias - lr * loss_wrt_bias


# Set np seed
np.random.seed(42)

# fmt: off
sample_inputs =  [[1, 3, 5],    [1, 9, 5],    [1, 2, 5],    [0, 9, 5],    [0, 0, 5]]
sample_outputs = [[1, 0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
# fmt: on

net = MyNetwork(3, 4)

lr = 0.01

for epoch_idx in range(2000):
    total_loss = 0
    for np_x, np_y in zip(sample_inputs, sample_outputs):
        np_x = np.array(np_x).astype(np.float32)
        np_y = np.array(np_y).astype(np.float32)
        total_loss += net.forward_loss(np_x, np_y)
        net.update_weights(np_x, np_y, lr)

    print(f"Epoch: {epoch_idx} Loss: {total_loss}")
