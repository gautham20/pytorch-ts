""" Depending on the version of the python and platform used an error
    may be raised due to the size of the pickle being loaded

    For context: https://stackoverflow.com/questions/31468117/python-3-can-pickle-handle-byte-objects-larger-than-4gb

"""
import pickle
from sys import platform


class MacOSFile(object):
    def __init__(self, f):
        self.f = f

    def __getattr__(self, item):
        return getattr(self.f, item)

    def read(self, n):
        # print("reading total_bytes=%s" % n, flush=True)
        if n >= (1 << 31):
            buffer = bytearray(n)
            idx = 0
            while idx < n:
                batch_size = min(n - idx, 1 << 31 - 1)
                # print("reading bytes [%s,%s)..." % (idx, idx + batch_size), end="", flush=True)
                buffer[idx : idx + batch_size] = self.f.read(batch_size)
                # print("done.", flush=True)
                idx += batch_size
            return buffer
        return self.f.read(n)

    def write(self, buffer):
        n = len(buffer)
        print("writing total_bytes=%s..." % n, flush=True)
        idx = 0
        while idx < n:
            batch_size = min(n - idx, 1 << 31 - 1)
            print(
                "writing bytes [{}, {})... ".format(idx, idx + batch_size),
                end="",
                flush=True,
            )
            self.f.write(buffer[idx : idx + batch_size])
            print("done.", flush=True)
            idx += batch_size


def pickle_load(file_path):
    with open(file_path, "rb") as f:
        if platform == "darwin":
            return pickle.load(MacOSFile(f))
        else:
            return pickle.load(f)


def pickle_dump(obj, file_path):
    with open(file_path, "wb") as f:
        if platform == "darwin":
            pickle.dump(obj, MacOSFile(f), protocol=pickle.HIGHEST_PROTOCOL)
        else:
            pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
