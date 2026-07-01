import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
import torch

import sys
# print(sys.executable)
import sys
sys.path.append("..")
import argparse
from datetime import datetime
import yaml
from utils.utils import *
from fl_utils.helper import Helper
from fl_utils.mnist_helper import Mnist_Helper
from fl_utils.cifar100_helper import Cifar100_Helper
from fl_utils.gtsrb_helper import GTSRB_Helper
from fl_utils.fashion_helper import FaMnist_Helper
from fl_utils.imagenet_helper import ImageNet_Helper
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

    parser = argparse.ArgumentParser(description="联邦学习鲁棒性实验核心控制台")
    parser.add_argument('--params', default='./yamls/common.yaml', help="配置文件路径")
    parser.add_argument('--gpu', default='0', help="显卡设备号 (若设置了CUDA_VISIBLE_DEVICES，此处代表局部逻辑卡号)")
    parser.add_argument('--data_folder', default='../data/', help="数据集存放根目录")
    parser.add_argument('--comment', default='', help="实验批次简短注释，用于独一化输出目录")
    args = parser.parse_args()

    if torch.cuda.is_available():
        # 检查是否通过环境变量限制了可见显卡 (例如 CUDA_VISIBLE_DEVICES=4)
        env_visible_gpus = os.environ.get("CUDA_VISIBLE_DEVICES")
        
        try:
            target_gpu = int(args.gpu)
            if env_visible_gpus is not None and "," not in env_visible_gpus:
                print(f">>> [GPU 提示] 检测到外部已锁定独立物理显卡 (GPU {env_visible_gpus})，内部自动对齐至局部逻辑卡号: cuda:0")
                torch.cuda.set_device(0)
                actual_gpu_id = 0
            else:
                if target_gpu < torch.cuda.device_count():
                    torch.cuda.set_device(target_gpu)
                    actual_gpu_id = target_gpu
                    print(f">>> [GPU 提示] 成功绑定目标显卡: cuda:{actual_gpu_id}")
                else:
                    print(f">>> [GPU 警告] 指定的 GPU {target_gpu} 超过可用范围，自动回滚至默认显卡: cuda:0")
                    torch.cuda.set_device(0)
                    actual_gpu_id = 0
        except Exception as e:
            print(f">>> [GPU 错误] 初始化 cuda set_device 异常 ({e})，采用系统默认驱动分配。")
            torch.cuda.set_device(0)
            actual_gpu_id = 0
    else:
        print(">>> [GPU 警告] 当前环境未检测到可用 CUDA 设备，程序将尝试在 CPU 上空转！")
        actual_gpu_id = 'cpu'

    if not os.path.exists(args.params):
        print(f"❌ [CRITICAL] 找不到指定的参数配置文件: {args.params}")
        sys.exit(1)

    with open(args.params, 'r', encoding='utf-8') as f:
        params = yaml.load(f, Loader=yaml.FullLoader)

    params['gpu'] = actual_gpu_id
    params['data_folder'] = args.data_folder
    params['comment'] = args.comment
    params['current_time'] = datetime.now().strftime('%b.%d_%H.%M.%S')

    logger.warning(create_table(params))
    
    if params['dataset'] == "cifar10":
        helper = Helper(params)
    elif params['dataset'] == "cifar100":
        helper = Cifar100_Helper(params)
    elif params['dataset'] == "tiny-imagenet-200":
        helper = ImageNet_Helper(params)
    elif params['dataset'] == "mnist":
        helper = Mnist_Helper(params)
    elif params['dataset'] == "gtsrb":
        helper = GTSRB_Helper(params)
    elif params['dataset'] == "emnist":
        helper = Emnist_Helper(params)
    elif params['dataset'] == "fashion-mnist":
        helper = FaMnist_Helper(params)
    main(helper)