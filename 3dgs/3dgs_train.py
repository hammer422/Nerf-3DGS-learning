import sys
import random
import numpy as np
import torch
import datetime



def safe_state(silent):
    old_f = sys.stdout
    class F:
        def __init__(self, silent):
            self.silent = silent

        def write(self, x):
            if not self.silent:
                if x.endswith("\n"):
                    old_f.write(x.replace("\n", " [{}]\n".format(str(datetime.now().strftime("%d/%m %H:%M:%S")))))
                else:
                    old_f.write(x)

        def flush(self):
            old_f.flush()

    sys.stdout = F(silent)

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.set_device(torch.device("cuda:0"))


class Config:
    # common params
    quiet = False
    detect_anomaly = False



    sh_degree = 3
    white_background = True

    iterations = 7000






def train():
    pass



    loss_for_log = 0.0





if __name__ == '__main__':
    args = Config()

    safe_state(args.quiet)

    torch.autograd.set_detect_anomaly(args.detect_anomaly)


    train()



