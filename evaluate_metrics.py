#EVALUATE

import numpy as np

def evaluate_predictions(gt, pred):
    
    pred_ori = pred
    gt_ori = gt

    pred[pred<=0] = 0.00001
    gt[gt<=0] = 0.00001

    thresh = np.maximum((gt_ori / pred_ori), (pred_ori / gt_ori))
        
    a1 = (thresh < 1.25   ).mean()
    a2 = (thresh < 1.25 ** 2).mean()
    a3 = (thresh < 1.25 ** 3).mean()

    abs_rel = np.mean(np.abs(gt_ori - pred_ori) / gt_ori)

    rmse = (gt_ori - pred_ori) ** 2
    rmse = np.sqrt(rmse.mean())

    return a1, a2, a3, abs_rel, rmse


true_depth = 'act_depth.npy'
pred_depth = 'pred_depth.npy'
	
a1, a2, a3, abs_rel, rmse = evaluate_predictions(true_depth,pred_depth)

print(sum_a1, sum_a2, sum_a3, sum_abs_rel, sum_rmse)
