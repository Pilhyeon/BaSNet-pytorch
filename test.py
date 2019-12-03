import torch
import torch.nn as nn
import numpy as np
import utils
import os
import json
from eval.eval_detection import ANETdetection
from tqdm import tqdm

def test(net, config, logger, test_loader, test_info, step, model_file=None):
    with torch.no_grad():
        net.eval()

        if model_file is not None:
            net.load_state_dict(torch.load(model_file))

        final_res = {}
        final_res['version'] = 'VERSION 1.3'
        final_res['results'] = {}
        final_res['external_data'] = {'used': True, 'details': 'Features from I3D Network'}

        num_correct = 0.
        num_total = 0.

        load_iter = iter(test_loader)
        
        for i in range(len(test_loader.dataset)):

            _data, _label, _, vid_name, vid_num_seg = next(load_iter)

            _data = _data.cuda()
            _label = _label.cuda()

            _, cas_base, score_supp, cas_supp, fore_weights = net(_data)

            label_np = _label.cpu().numpy()
            score_np = score_supp[0,:-1].cpu().data.numpy()

            score_np[np.where(score_np < config.class_thresh)] = 0
            score_np[np.where(score_np >= config.class_thresh)] = 1

            correct_pred = np.sum(label_np == score_np, axis=1)

            num_correct += np.sum((correct_pred == config.num_classes).astype(np.float32))
            num_total += correct_pred.shape[0]
            
            cas_base = utils.minmax_norm(cas_base)
            cas_supp = utils.minmax_norm(cas_supp)

            pred = np.where(score_np > config.class_thresh)[0]

            if pred.any():
                cas_pred = cas_supp[0].cpu().numpy()[:, pred]
                cas_pred = np.reshape(cas_pred, (config.num_segments, -1, 1))

                cas_pred = utils.upgrade_resolution(cas_pred, config.scale)
                
                proposal_dict = {}

                for i in range(len(config.act_thresh)):
                    cas_temp = cas_pred.copy()

                    zero_location = np.where(cas_temp[:, :, 0] < config.act_thresh[i])
                    cas_temp[zero_location] = 0

                    seg_list = []
                    for c in range(len(pred)):
                        pos = np.where(cas_temp[:, c, 0] > 0)
                        seg_list.append(pos)

                    proposals = utils.get_proposal_oic(seg_list, cas_temp, score_np, pred, config.scale, \
                                    vid_num_seg[0].cpu().item(), config.feature_fps, config.num_segments)

                    for i in range(len(proposals)):
                        class_id = proposals[i][0][0]

                        if class_id not in proposal_dict.keys():
                            proposal_dict[class_id] = []

                        proposal_dict[class_id] += proposals[i]

                final_proposals = []
                for class_id in proposal_dict.keys():
                    final_proposals.append(utils.nms(proposal_dict[class_id], 0.7))

                final_res['results'][vid_name[0]] = utils.result2json(final_proposals)

        test_acc = num_correct / num_total

        json_path = os.path.join(config.output_path, 'temp_result.json')
        with open(json_path, 'w') as f:
            json.dump(final_res, f)
            f.close()

        tIoU_thresh = np.linspace(0.1, 0.9, 9)
        anet_detection = ANETdetection(config.gt_path, json_path,
                                   subset='test', tiou_thresholds=tIoU_thresh,
                                   verbose=False, check_status=False)
        mAP, average_mAP = anet_detection.evaluate()

        logger.log_value('Test accuracy', test_acc, step)

        for i in range(tIoU_thresh.shape[0]):
            logger.log_value('mAP@{:.1f}'.format(tIoU_thresh[i]), mAP[i], step)

        logger.log_value('Average mAP', average_mAP, step)

        test_info["step"].append(step)
        test_info["test_acc"].append(test_acc)
        test_info["average_mAP"].append(average_mAP)

        for i in range(tIoU_thresh.shape[0]):
            test_info["mAP@{:.1f}".format(tIoU_thresh[i])].append(mAP[i])
