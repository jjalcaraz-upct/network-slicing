import numpy as np

class SVvariable:
    def __init__(self):
        self.counter = 0

    def extend(self, new_coeff):
        if self.counter == 0:
            self.coeff = np.array([new_coeff], dtype=np.float32)
        else:
            self.coeff = np.append(self.coeff , new_coeff)

    def update(self, new_coeff):
        self.coeff += new_coeff

    def insert(self, x):
        if self.counter == 0:
            self.landmarks = x
        else:
            self.landmarks = np.vstack((self.landmarks, x))
        self.counter += 1

class Projectron:
    # kernel object must use SVvariable
    def __init__(self, kernel, eta = 0.1):
        self.kernel = kernel
        self.sv = self.kernel.sv
        self.eta = eta
        self.Kinv = np.array([0.0], dtype=np.float32)
        self.counter = 0

    def predict(self, x):
        if self.counter > 0:
            y_pred, self.f, self.K_f = self.kernel.predict(x)
        else:
            y_pred, self.f, self.K_f = 0, 0.0, np.array([0.0], dtype=np.float32)
        return y_pred

    def update(self, x, y):
        if self.f * y <= 0:
            Kii = self.kernel.k_eval(x,x)
            d_star = self.Kinv @ self.K_f
            if np.ndim(d_star) == 0:
                d_star = np.array([d_star], dtype=np.float32)
            delta = max(Kii - d_star @ self.K_f, 0)

            if delta <= self.eta:
                self.sv.update(y * d_star)
            else:
                self.sv.extend(y)
                self.sv.insert(x)
                self.counter += 1

                if self.counter > 1:
                    self.Kinv = np.vstack((self.Kinv, np.zeros(self.counter-1)))
                    self.Kinv = np.column_stack((self.Kinv, np.zeros(self.counter)))
                    d_star_extend =  np.append(d_star, -1)
                    self.Kinv += np.outer(d_star_extend,d_star_extend) /delta
                else:
                    self.Kinv = np.array([1.0/Kii], dtype=np.float32)

    def get_set_size(self): 
        # for complexity evaluation only
        return self.kernel.sv.landmarks.shape[0]

class ProjectronPlus(Projectron):
    # kernel object must use SVvariable
    def __init__(self, kernel, eta = 0.1):
        super().__init__(kernel, eta)

    def update(self, x, y):
        margin = y * self.f

        if margin < 1 and margin > 0:
            loss = (1 - margin)
            Kii = self.kernel.k_eval(x,x)
            d_star = self.Kinv @ self.K_f
            if np.ndim(d_star) == 0:
                d_star = np.array([d_star], dtype=np.float32)
            delta = max(Kii - d_star @ self.K_f, 0)
            norm_xt = max(Kii - delta, 0)

            if loss - delta/self.eta > 0:
                alpha = min(min(loss/norm_xt, 1), 2*(loss - delta/self.eta)/norm_xt)
                self.sv.update(alpha * y * d_star)

        elif margin <= 0:
            Kii = self.kernel.k_eval(x,x)
            d_star = self.Kinv @ self.K_f
            if np.ndim(d_star) == 0:
                d_star = np.array([d_star], dtype=np.float32)
            delta = max(Kii - d_star @ self.K_f, 0)

            if delta <= self.eta:
                self.sv.update(y * d_star)
            else:
                self.sv.extend(y)
                self.sv.insert(x)
                self.counter += 1

                if self.counter > 1:
                    self.Kinv = np.vstack((self.Kinv, np.zeros(self.counter-1)))
                    self.Kinv = np.column_stack((self.Kinv, np.zeros(self.counter)))
                    d_star_extend =  np.append(d_star, -1)
                    self.Kinv += np.outer(d_star_extend,d_star_extend) /delta
                else:
                    self.Kinv = np.array([1.0/Kii], dtype=np.float32)


if __name__ == '__main__':
    from kernel import GaussianKernel
    DIM = 3
    SAMPLES = 20000
    sv = SVvariable(DIM)
    kernel = GaussianKernel(sv, 1)
    algorithm = ProjectronPlus(kernel)
    correct = 0
    for i in range(SAMPLES):
        x = np.random.rand(DIM)
        y_pred = algorithm.predict(x)
        y = -1
        if x.sum() > 1.5:
            y = 1
        algorithm.update(x, y)
        if y_pred == y:
            correct += 1
        if i % 100 == 0:
            success_rate = correct/(i+1)
            print('{}: success rate = {}'.format(i, success_rate))
