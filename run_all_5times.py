# nohup python3 train.py --protocol O_C_M_to_I > ocmi_noSeed_1.out &
import datetime as dt
import os
import argparse


parser = argparse.ArgumentParser(
        prog='run_all_5times',
        description='run each protocol 5 times')
parser.add_argument('--exchange', type=str, default='',
                    choices=['patchexchange', 'landmarkexchange'])
parser.add_argument('--times', type=int, default=5)
parser.add_argument('--output_dir', type=str, required=True)
args = parser.parse_args()


def raises_error(n, p, i):
    cmd = f"python3 train.py --protocol {p}"
    if p == "O1":
        cmd += " --data_dir /home/rgpa18/ssan_datasets/original/oulu-npu"
    if len(args.exchange) > 0:
        cmd += f" --exchange_aug {args.exchange}"
    out_file = f"{args.output_dir}/{n}_{args.exchange}{'_'*len(args.exchange)}{i}.out"
    time_str = dt.datetime.now().strftime("%H:%M:%S:%f")
    prefix = f"echo {time_str}"
    return os.system(f"{prefix} && {cmd} > {out_file}")


names = ["mrc", "mro", "ocmr", "ocrm", "omrc", "rcmo"]
protocols = ["M_I_to_C", "M_I_to_O", "O_C_M_to_I", "O_C_I_to_M", "O_M_I_to_C",
             "I_C_M_to_O"]

for n, p in zip(names, protocols):
    for i in range(args.times):
        print(f"Now running protocol {n} ({p}) for the {i}-th time")
        while raises_error(n, p, i):
            continue
