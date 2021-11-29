from complex_fc_cpu import FullyConnected
from PowerLayer import Power
from SoftmaxLayer import Softmax
from ArgmaxLayer import Argmax
from LossLayer import CrossEntroy


class Model:
    def __init__(self, num_beams, num_ant, batch_size, mode='orig', accum=False):
        # layers
        self.ComplexFC = FullyConnected(num_beams, num_ant, batch_size, mode, accum)
        self.Power = Power(num_beams)
        self.SoftMax = Softmax(num_beams)
        self.ArgMax = Argmax(num_beams)
        self.Loss = CrossEntroy(batch_size)
        # codebook and gradient
        self.batch_size = batch_size
        self.codebook = self.ComplexFC.thetas
        self.grad = self.ComplexFC.grad

    def forward(self, h, val_mode=False, val_size=100):
        A = self.ComplexFC.forward(h, val_mode)
        S = self.Power.forward(A, val_mode)
        P = self.SoftMax.forward(S, val_mode)
        L = self.ArgMax.forward(S, val_mode)
        loss = self.Loss.forward(P, L, val_mode, val_size)
        return loss

    def backward(self):
        dL_dP = self.Loss.backward()
        dL_dS = self.SoftMax.backward(dL_dP)
        dL_dA = self.Power.backward(dL_dS)
        dydx = self.ComplexFC.backward(dL_dA)
        self.grad = dydx
        return dydx

    def update(self, lr=0.1):
        self.codebook = self.ComplexFC.update(lr=lr)
        return self.codebook
