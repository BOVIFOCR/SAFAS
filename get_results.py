import os
import numpy as np
import sys
from pathlib import Path


class Table:
    UNKNOWN_LEN = -1
    def __init__(self, headers=None):
        self.headers = headers
        self.row_len = self.UNKNOWN_LEN
        self.rows = []

    def add_row(self, row_data):
        assert self.row_len == self.UNKNOWN_LEN or self.row_len == len(row_data)
        if self.row_len == self.UNKNOWN_LEN:
            self.init_row_len(len(row_data))
        self.rows.append(row_data)

    def init_row_len(self, row_len):
        self.row_len = row_len
        if self.headers is None:
            self.headers = list(map(lambda x: f'col_{x}', range(self.row_len)))

    def render_markdown(self):
        print(' | '.join(self.headers))
        print('|'.join(['---']*self.row_len))
        for row in self.rows:
            print(' | '.join(map(str, row)))


def old_way():
    """
    do it the stupid way, files have to follow a name structure and you
    have to know how many of each protocol there are
    """
    # names = ["mrc", "mro", "ocmr", "ocrm", "omrc", "rcmo"]
    names = ["mrc", "mro", "ocm2i", "oci2m", "omi2c", "icm2o"]
    protocols = ["M_I_to_C", "M_I_to_O", "O_C_M_to_I", "O_C_I_to_M", "O_M_I_to_C",
                 "I_C_M_to_O"]
    d = sys.argv[1]

    table = Table(['protocol', 'mean', 'min'])
    for n in names:
        os.system("echo > tmp.txt")
        for i in range(5):
            os.system((f'grep "Best" {d}/'
                       + f'{n}_noSeed_noExtendedTrainSet_{i}.out'
                       + ' | tail -n 1 | grep -oE "HTER=([0-9]|\\.)+"'
                       + ' | grep -oE "([0-9]|\\.)+" >> tmp.txt'))
        with open("tmp.txt", 'r') as f:
            lines = [x.strip() for x in f.readlines()]
        lines = np.array(list(map(float, filter(lambda x: len(x), lines))))
        mean = lines.mean()*100
        std = lines.std()*100
        mn = lines.min()*100
        mx = lines.max()*100
        table.add_row([n.upper(), f'{mean:.2f}', f'{mn:.2f}'])
    table.render_markdown()


def get_lcp(strings):
    """get lcp of a list of strings"""
    n = min(len(s) for s in strings)
    last = list(strings[0][:n])
    r = n
    for s in strings[1:]:
        for p in range(r):
            if s[p] != last[p]:
                r = p
                break
    return r


def smart_way():
    """remove lcp of all filenames, remove lcs of all filenames, then work with that"""
    directory = Path(sys.argv[1])
    files = list(directory.iterdir())
    filenames = sorted(list(map(lambda x: x.name, files)))
    lcp_len = get_lcp(filenames)
    lcs_len = 4  # .out

    table = Table(['protocol', 'mean', 'min'])

old_way()
