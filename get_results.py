import os
import numpy as np
import sys


names = ["mrc", "mro", "ocmr", "ocrm", "omrc", "rcmo"]
protocols = ["M_I_to_C", "M_I_to_O", "O_C_M_to_I", "O_C_I_to_M", "O_M_I_to_C",
             "I_C_M_to_O"]
d = sys.argv[1]

print("protocol | mean | std | min | max")
print("---|---|---|---")
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
    print(f"{n.upper()} | {mean:.2f} | {std:.2f} | {mn:.2f} | {mx:.2f}")
