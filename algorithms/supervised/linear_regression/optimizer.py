class GradientDescent:
    def __init__(self, lr=0.01):
        self.lr = lr

    def step(self, w, b, dw, db):
        """
        Updates parameters using gradient descent.
        """
        w = w - self.lr * dw
        b = b - self.lr * db
        return w, b
