import numpy as np

class GaussianKernel:
    def __init__(self, sv, gamma = 1.0):
        self.sv = sv
        self.gamma = gamma

    def k_eval(self, x1, x2):
        dist = (x1 - x2)**2
        k = np.exp(-self.gamma*dist.sum())
        return k

    def k(self, x):
        l = self.sv.landmarks
        if np.ndim(l) == 1:
            k = np.array([self.k_eval(l, x)], dtype=np.float32)
        else:
            dist = (l - x)**2
            k = np.exp(-self.gamma*dist.sum(axis=1))
        return k

    def predict(self, x):
        k = self.k(x)
        f = k @ self.sv.coeff
        y = np.sign(f)
        if y == 0:
            y = np.random.choice([-1,1])
        return y, f, k

    def update(self, rate, new_coeff):
        self.sv.update(rate, new_coeff)

    def insert(self, x):
        self.sv.insert(x)

class SV:
    def __init__(self, dimension, budget):
        self.landmarks = np.zeros((budget, dimension), dtype=np.float32)
        self.counter = 0
        self.budget = budget
        self.coeff = np.zeros((budget), dtype=np.float32)

    def update(self, rate, new_coeff):
        self.coeff = rate*self.coeff
        self.coeff[self.counter] = new_coeff

    def insert(self, x):
        self.landmarks[self.counter,:] = x
        self.counter += 1
        self.counter = self.counter % self.budget

if __name__ == '__main__':
    sv = SV(3,5)
    kernel = GaussianKernel(sv)
    for i in range(10):
        x = np.full((3),i, dtype=np.float32)
        k = kernel.k(x)
        y, _ = kernel.predict(x)
        sv.insert(x)
        print('x = {}'.format(x))
        print('y = {}'.format(y))
        print('k = {}'.format(k))
