class GradientDescent:
    def __init__(self, lr=0.01):
        self.lr = lr

    def step(self, w, b, dw, db):
        w -= self.lr * dw
        b -= self.lr * db
        return w, b
