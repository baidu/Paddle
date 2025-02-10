import paddle
paddle.set_device('gpu:0')
gt_labels_3d = paddle.load("./gt_labels_3d.pdparams", return_numpy=False)
valid_mask = paddle.load("./valid_mask.pdparams", return_numpy=False)
mask = paddle.load("./mask.pdparams", return_numpy=False)
center_y = paddle.load("./center_y.pdparams", return_numpy=False)
center_x = paddle.load("./center_x.pdparams", return_numpy=False)
gt_heatmaps_validflag = paddle.load("./gt_heatmaps_validflag.pdparams", return_numpy=False)

print("valid_mask = ", valid_mask, flush=True)
tmp = gt_labels_3d[mask][valid_mask]
print("tmp = ", tmp, flush=True)
tmp2 = center_y[valid_mask].cast('long')
print("tmp2 = ", tmp2, flush=True)
tmp3 = center_x[valid_mask].cast('long')
print("tmp3 = ", tmp3, flush=True)
gt_heatmaps_validflag[tmp, tmp2, tmp3] = 0

gt_heatmaps_validflag[gt_labels_3d[mask][valid_mask], center_y[valid_mask].cast('long'), center_x[valid_mask].cast('long')] = 0
