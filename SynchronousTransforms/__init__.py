"""
This class is implied for the reasons:
    1. The transform in torchvision can't guarantee that the input and ground truth has the same transformation
    2. The data is not a PIL image
"""


class Rand:
    def __init__(self, seed=1234):
        self.seed = seed
        self.m = 2 ^ 31
        self.a = 1103515245
        self.c = 12345

        self.x_n = seed
        self.x_n_backup = seed  # record the x_n value before reset

    def step(self):
        self.x_n = self.x_n_backup

    def rand(self, num=1):
        x_n = self.x_n
        result = []
        for i in range(num):
            self.x_n = (self.a * self.x_n + self.c) % self.m
            result.append(self.x_n / self.m)
        self.x_n_backup = self.x_n
        self.x_n = x_n
        return result if num > 1 else result[0]
