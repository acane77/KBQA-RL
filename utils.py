import pickle
import os

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

