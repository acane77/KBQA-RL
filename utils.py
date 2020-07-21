import pickle
import os

def save_binary(filename, data):
    pickle.dump(data, open("./results/binaries/" + filename, "wb"))

def load_binary(filename):
    return pickle.load(open("./results/binaries/" + filename, "rb"))

def check_binary(filename):
    return os.path.isfile("./results/binaries/" + filename)

