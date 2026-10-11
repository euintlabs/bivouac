import numpy as np


class MLP:
    def __init__(self, nin=11, hidden=8, nout=5):
        self.nin, self.hidden, self.nout = nin, hidden, nout
        self.nparam = hidden * nin + hidden + nout * hidden + nout

    def unpack(self, th):
        i = 0
        W1 = th[i:i + self.hidden * self.nin].reshape(self.hidden, self.nin); i += self.hidden * self.nin
        b1 = th[i:i + self.hidden]; i += self.hidden
        W2 = th[i:i + self.nout * self.hidden].reshape(self.nout, self.hidden); i += self.nout * self.hidden
        b2 = th[i:i + self.nout]
        return W1, b1, W2, b2

    def forward(self, th, obs):
        W1, b1, W2, b2 = self.unpack(th)
        h = np.tanh(obs @ W1.T + b1)
        return np.tanh(h @ W2.T + b2)

    def export(self, th, digits=5):
        W1, b1, W2, b2 = self.unpack(th)
        r = lambda a: [round(float(v), digits) for v in a]
        return {"W1": [r(row) for row in W1], "b1": r(b1), "W2": [r(row) for row in W2], "b2": r(b2)}
