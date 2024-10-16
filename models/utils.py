import collections
import torch
import torch.nn.functional as F

Rays = collections.namedtuple("Rays", ("origins", "viewdirs"))

def namedtuple_map(fn, tup):
    """Apply `fn` to each element of `tup` and cast to `tup`'s namedtuple."""
    return type(tup)(*(None if x is None else fn(x) for x in tup))

def collect_img_pts(pts, sample_idx):
    bs, sample_num, _ = sample_idx.shape
    batch_inds = torch.arange(bs).reshape(-1, 1).repeat(1, sample_num).reshape(-1).cuda()
    pts_reshape = pts.permute(0, 2, 3, 1)
    idx_reshape = sample_idx.reshape(-1, 2)
    pts_sel = pts_reshape[batch_inds.view(-1), idx_reshape[:, 0], idx_reshape[:, 1]].reshape(bs, sample_num, -1)
    return pts_sel

def collect_pts_feat(pts, idx):
    pts_reshape = pts.permute(0, 2, 1)
    bs, sample_num, _ = idx.shape
    batch_inds = torch.arange(bs).reshape(-1, 1).repeat(1, sample_num).reshape(-1).cuda()
    idx_reshape = idx.view(-1)
    pts_sel = pts_reshape[batch_inds, idx_reshape].reshape(bs, sample_num, -1)
    return pts_sel

def geodesic_distance(pred_R, gt_R):
    '''
    q is the output from the network (rotation from t=0.5 to t=1)
    gt_R is the GT rotation from t=0 to t=1
    '''
    pred_R, gt_R = pred_R.cpu(), gt_R.cpu()
    R_diff = torch.matmul(pred_R, gt_R.T)
    cos_angle = torch.clip((torch.trace(R_diff) - 1.0) * 0.5, min=-1., max=1.)
    angle = torch.rad2deg(torch.arccos(cos_angle)) 
    angle = torch.nan_to_num(angle, nan=0)
    return angle


def axis_metrics(motion, gt):
    # pred axis
    pred_axis_d = motion['axis_d'].cpu().squeeze(0)
    pred_axis_o = motion['axis_o'].cpu().squeeze(0)
    # gt axis
    gt_axis_d = gt['axis_d']
    gt_axis_o = gt['axis_o']
    # angular difference between two vectors
    cos_theta = torch.dot(pred_axis_d, gt_axis_d) / (torch.norm(pred_axis_d) * torch.norm(gt_axis_d))
    ang_err = torch.rad2deg(torch.acos(torch.abs(cos_theta)))
    # positonal difference between two axis lines
    w = gt_axis_o - pred_axis_o
    cross = torch.cross(pred_axis_d, gt_axis_d)
    if (cross == torch.zeros(3)).sum().item() == 3:
        pos_err = torch.tensor(0)
    else:
        pos_err = torch.abs(torch.sum(w * cross)) / torch.norm(cross)
    return ang_err, pos_err

def translational_error(motion, gt):
    dist = motion['dist'].cpu()
    # dist = dist_half * 2.
    gt_dist = gt['dist']

    axis_d = F.normalize(motion['axis_d'].cpu().squeeze(0), p=2, dim=0)
    gt_axis_d = F.normalize(gt['axis_d'].cpu(), p=2, dim=0)

    err = torch.sqrt(((dist * axis_d - gt_dist * gt_axis_d) ** 2).sum())
    return err

def entropy_loss(src, clip_eps=1e-6, skew=1.0):
    """
    "skew" is used to control the skew of entropy loss.
    """
    src = torch.clip(src ** skew, clip_eps, 1-clip_eps)
    entropy = - (src * torch.log(src) + (1.-src) * torch.log(1.-src))
    return entropy.mean()