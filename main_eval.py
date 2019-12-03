import pdb
import numpy as np
import torch.utils.data as data
import utils
from options import *
from config import *
from test import *
from model import *
from tensorboard_logger import Logger
from thumos_features import *


if __name__ == "__main__":
    args = parse_args()
    if args.debug:
        pdb.set_trace()

    config = Config(args)
    worker_init_fn = None

    if config.seed >= 0:
        utils.set_seed(config.seed)
        worker_init_fn = np.random.seed(config.seed)

    net = BaS_Net(config.len_feature, config.num_classes, config.num_segments)
    net = net.cuda()

    test_loader = data.DataLoader(
        ThumosFeature(data_path=config.data_path, mode='test',
                        modal=config.modal, feature_fps=config.feature_fps,
                        num_segments=config.num_segments, len_feature=config.len_feature,
                        seed=config.seed, sampling='uniform'),
            batch_size=1,
            shuffle=False, num_workers=config.num_workers,
            worker_init_fn=worker_init_fn)

    test_info = {"step": [], "test_acc": [], "average_mAP": [],
                "mAP@0.1": [], "mAP@0.2": [], "mAP@0.3": [],
                "mAP@0.4": [], "mAP@0.5": [], "mAP@0.6": [],
                "mAP@0.7": [], "mAP@0.8": [], "mAP@0.9": []}
    
    logger = Logger(config.log_path)
        
    test(net, config, logger, test_loader, test_info, 0, model_file=config.model_file)

    utils.save_best_record_thumos(test_info, 
        os.path.join(config.output_path, "best_record.txt".format(config.seed)))