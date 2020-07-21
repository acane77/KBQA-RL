import pickle
import os
import torch
from expeiment_settings import ExperimentSettings

class Utility:
    class Binary:
        @staticmethod
        def save(filename: str, data):
            pickle.dump(data, open("./results/binaries/" + filename, "wb"))

        @staticmethod
        def load(filename: str):
            return pickle.load(open("./results/binaries/" + filename, "rb"))

        @staticmethod
        def exists(filename: str):
            return os.path.isfile("./results/binaries/" + filename)

    @staticmethod
    def to_gpu(x):
        if ExperimentSettings.enable_cuda and torch.cuda.is_available():
            return x.cuda()
        return x
