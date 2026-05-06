import sys, os
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# print(sys.executable)
import sys
sys.path.append("..")
import argparse
from datetime import datetime
import yaml
from fl_utils.helper import Helper
from fl_utils.imagenet_helper import ImageNet_Helper
import logging
from fl_utils.helper import Helper
from fl_utils.mnist_helper import Mnist_Helper
from fl_utils.cifar100_helper import Cifar100_Helper
from fl_utils.fler import FLer
import warnings
import torch
import numpy as np
import random


warnings.filterwarnings('ignore')
logger = logging.getLogger('logger')

# 设置随机种子
# seed = int(time.time()) # 1234  42
# seed = 42
# seed = 1728394385

# cifar100 1722390053  1722348895
# seed = 1722390053
# seed = 1722348895

# cifar10 1722440506   1722467484
# seed = 1722440506

# seed = 1722467484

# imagenet
#seed = 1724224346
seed = 42

torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False


def main(helper):
    # helper = Helper(params)
    # print(f"本轮的随机种子为:{seed}")
    fler = FLer(helper)
    fler.train()
    # helper.remove_update()

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    #
    # parser.add_argument('--params', default='./yamls/cifar100.yaml')
    # parser.add_argument('--params', default='./yamls/imagenet.yaml')
    # parser.add_argument('--params', default='./yamls/tmp.yaml')
    parser.add_argument('--params', default='./yamls/cifar100_DOBA_nomia.yaml')
    parser.add_argument('--gpu', default=0)
    parser.add_argument('--data_folder', default='../data/')
    args = parser.parse_args()
    torch.cuda.set_device(int(args.gpu))

    with open(args.params) as f:
        params = yaml.load(f, Loader=yaml.FullLoader)

    params['current_time'] = datetime.now().strftime('%b.%d_%H.%M.%S')
    if params["dataset"] == "cifar10":
        helper = Helper(params)
    elif params["dataset"] == "cifar100":
        helper = Cifar100_Helper(params)
    elif params["dataset"] == "mnist":
        helper = Mnist_Helper(params)
    elif params["dataset"] == "imagenet":
        helper = ImageNet_Helper(params)
    else:
        print("No such Dataset")
        exit(1)
    main(helper)
