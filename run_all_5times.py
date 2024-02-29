# nohup python3 train.py --protocol O_C_M_to_I > ocmi_noSeed_1.out &
import datetime as dt
import os


def raises_error(n, p, i):
    cmd = f"python3 train.py --protocol {p}"
    out_file = f"output_files/{n}_noSeed_noExtendedTrainSet_{i}.out"
    time_str = dt.datetime.now().strftime("%H:%M:%S:%f")
    prefix = f"echo {time_str}"
    return os.system(f"{prefix} && {cmd} > {out_file}")


names = ["mrc", "mro", "ocmr", "ocrm", "omrc", "rcmo"]
protocols = ["M_I_to_C", "M_I_to_O", "O_C_M_to_I", "O_C_I_to_M", "O_M_I_to_C",
             "I_C_M_to_O"]

for n, p in zip(names, protocols):
    for i in range(5):
        print(f"Now running protocol {n} ({p}) for the {i}-th time")
        while raises_error(n, p, i):
            continue
