import pdb
import numpy as np
import torch.utils.data as data
import utils
from options import *
from config import *
from train import *
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

    train_loader = data.DataLoader(
        ThumosFeature(data_path=config.data_path, mode='train',
                        modal=config.modal, feature_fps=config.feature_fps,
                        num_segments=config.num_segments, len_feature=config.len_feature,
                        seed=config.seed, sampling='random'),
            batch_size=config.batch_size,
            shuffle=True, num_workers=config.num_workers,
            worker_init_fn=worker_init_fn)

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
    
    best_mAP = -1

    criterion = BaS_Net_loss(config.alpha)

    optimizer = torch.optim.Adam(net.parameters(), lr=config.lr[0],
        betas=(0.9, 0.999), weight_decay=0.0005)

    logger = Logger(config.log_path)
    
    loader_iter = iter(train_loader)

    for step in tqdm(
            range(1, config.num_iters + 1),
            total = config.num_iters,
            dynamic_ncols = True
        ):
        if step > 1 and config.lr[step - 1] != config.lr[step - 2]:
            for param_group in optimizer.param_groups:
                param_group["lr"] = config.lr[step - 1]

        train(net, train_loader, loader_iter, optimizer, criterion, logger, step)
        
        test(net, config, logger, test_loader, test_info, step)

        if test_info["average_mAP"][-1] > best_mAP:
            best_mAP = test_info["average_mAP"][-1]

            utils.save_best_record_thumos(test_info, 
                os.path.join(config.output_path, "best_record_seed_{}.txt".format(config.seed)))

            torch.save(net.state_dict(), os.path.join(args.model_path, \
                "BaS_Net_model_seed_{}.pkl".format(config.seed)))

