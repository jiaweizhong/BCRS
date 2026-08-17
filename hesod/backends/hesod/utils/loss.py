# Loss functions
# Copyright (c) Alibaba, Inc. and its affiliates.

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

from utils.general import bbox_iou, box_iou, wh_iou, xywh2xyxy
from utils.torch_utils import is_parallel, time_synchronized


def smooth_BCE(eps=0.1):  # https://github.com/ultralytics/yolov3/issues/238#issuecomment-598028441
    # return positive, negative label smoothing BCE targets
    return 1.0 - 0.5 * eps, 0.5 * eps


def sabl_loss(pbox, tbox, stride, scale=32.0, beta=6.0, normalizer=12.0, eps=1e-7):
    """SSABNet scale-adaptive bounding-box loss for matched xywh boxes.

    ``pbox`` and ``tbox`` use detection-grid coordinates. CIoU's scale-free
    terms are evaluated in that coordinate system, while the target scale and
    Wasserstein distance are converted to network-input pixels as required by
    SSABNet Equations (10)-(15). The returned tensor contains one loss value
    per matched box.
    """
    if scale <= 0 or beta <= 0 or normalizer <= 0:
        raise ValueError('SABL scale, beta, and normalizer must all be positive')
    if pbox.ndim != 2 or pbox.shape[-1] != 4 or pbox.shape != tbox.shape:
        raise ValueError('SABL expects pbox and tbox with matching shape (N, 4)')

    # The IoU, normalized center-distance, and aspect-ratio terms reproduce
    # this vendor's bbox_iou(..., CIoU=True) decomposition. Keeping them in
    # grid units avoids needless rescaling because all three are scale-free.
    pxy, pwh = pbox[:, :2], pbox[:, 2:4]
    txy, twh = tbox[:, :2], tbox[:, 2:4]
    p_half, t_half = pwh / 2, twh / 2
    p_min, p_max = pxy - p_half, pxy + p_half
    t_min, t_max = txy - t_half, txy + t_half

    inter = (torch.min(p_max, t_max) - torch.max(p_min, t_min)).clamp(min=0).prod(1)
    p_area = pwh[:, 0] * (pwh[:, 1] + eps)
    t_area = twh[:, 0] * (twh[:, 1] + eps)
    iou = inter / (p_area + t_area - inter + eps)

    enclosing_wh = torch.max(p_max, t_max) - torch.min(p_min, t_min)
    enclosing_diag_sq = enclosing_wh.square().sum(1) + eps
    center_distance_sq = (pxy - txy).square().sum(1)
    euclidean_penalty = center_distance_sq / enclosing_diag_sq

    v = (4 / math.pi ** 2) * torch.pow(
        torch.atan(twh[:, 0] / (twh[:, 1] + eps))
        - torch.atan(pwh[:, 0] / (pwh[:, 1] + eps)), 2
    )
    with torch.no_grad():
        alpha = v / (v - iou + (1 + eps))

    # W2^2 for the Gaussian box representation is the squared L2 distance
    # between (cx, cy, w/2, h/2). Its square root and the GT geometric-mean
    # scale must be in input pixels, not feature-grid cells.
    pixel_stride = torch.as_tensor(stride, device=pbox.device, dtype=pbox.dtype)
    wasserstein_delta = torch.cat((pxy - txy, (pwh - twh) / 2), 1) * pixel_stride
    wasserstein_distance = wasserstein_delta.norm(p=2, dim=1)
    wasserstein_penalty = 1.0 - torch.exp(-wasserstein_distance / normalizer)
    target_scale = torch.sqrt((twh[:, 0] * twh[:, 1]).clamp(min=0)) * pixel_stride
    small_object_weight = torch.exp(-torch.pow(target_scale / scale, beta))

    hybrid_penalty = (
        small_object_weight * wasserstein_penalty
        + (1.0 - small_object_weight) * euclidean_penalty
    )
    return 1.0 - iou + hybrid_penalty + alpha * v


class FixedTextureFilter(nn.Module):
    """Non-trainable Sobel-magnitude filter defining B_tex for F6's
    rescue-ranking/conditional-gate-regularization losses (HESOD-Agri-
    Proposal.md SS4.2.2). Runs on the raw input image, NOT on the model's own
    learned t_bg output or any other learned quantity -- t_bg is itself being
    trained via L_cond's penalty on B_tex membership, so B_tex must not
    depend on it (would be circular: the target would move with the model).

    Mirrors the offline `tfr_diagnose.py` diagnostic's Sobel-magnitude
    definition, but per-batch rather than a full-test-set pass (a training
    batch's background-pixel population is far smaller/noisier than a full
    dataset pass, so the quantile threshold is computed fresh per batch, not
    reused as a fixed constant).
    """
    def __init__(self):
        super().__init__()
        sobel_x = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]])
        sobel_y = torch.tensor([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]])
        weight = torch.stack([sobel_x, sobel_y], dim=0).unsqueeze(1)  # (2,1,3,3)
        self.conv = nn.Conv2d(1, 2, kernel_size=3, padding=1, bias=False)
        self.conv.weight = nn.Parameter(weight, requires_grad=False)

    @torch.no_grad()
    def forward(self, gray):  # gray: (bs,1,H,W), full input resolution
        g = self.conv(gray)
        return torch.sqrt(g[:, 0:1].pow(2) + g[:, 1:2].pow(2) + 1e-12)  # (bs,1,H,W)


class BCEBlurWithLogitsLoss(nn.Module):
    # BCEwithLogitLoss() with reduced missing label effects.
    def __init__(self, alpha=0.05):
        super(BCEBlurWithLogitsLoss, self).__init__()
        self.loss_fcn = nn.BCEWithLogitsLoss(reduction='none')  # must be nn.BCEWithLogitsLoss()
        self.alpha = alpha

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)
        pred = torch.sigmoid(pred)  # prob from logits
        dx = pred - true  # reduce only missing label effects
        # dx = (pred - true).abs()  # reduce missing label and false label effects
        alpha_factor = 1 - torch.exp((dx - 1) / (self.alpha + 1e-4))
        loss *= alpha_factor
        return loss.mean()


class FocalLoss(nn.Module):
    # Wraps focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)
    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        super(FocalLoss, self).__init__()
        self.loss_fcn = loss_fcn  # must be nn.BCEWithLogitsLoss()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = 'none'  # required to apply FL to each element

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)
        # p_t = torch.exp(-loss)
        # loss *= self.alpha * (1.000001 - p_t) ** self.gamma  # non-zero power for gradient stability

        # TF implementation https://github.com/tensorflow/addons/blob/v0.7.1/tensorflow_addons/losses/focal_loss.py
        pred_prob = torch.sigmoid(pred)  # prob from logits
        p_t = true * pred_prob + (1 - true) * (1 - pred_prob)
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = (1.0 - p_t) ** self.gamma
        loss *= alpha_factor * modulating_factor

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss


class QFocalLoss(nn.Module):
    # Wraps Quality focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)
    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        super(QFocalLoss, self).__init__()
        self.loss_fcn = loss_fcn  # must be nn.BCEWithLogitsLoss()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = 'none'  # required to apply FL to each element

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)

        pred_prob = torch.sigmoid(pred)  # prob from logits
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = torch.abs(true - pred_prob) ** self.gamma
        loss *= alpha_factor * modulating_factor

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss


class ComputeLoss:
    # Compute losses
    def __init__(self, model, autobalance=False, selector_loss='upstream', lambda_cov=0.5, pos_weight=2.0,
                 box_loss='upstream', box_weight_ref_area=4.0, box_weight_max=5.0,
                 lambda_rescue=0.0, lambda_cond=0.0, tau_low=0.3, tau_high=None,
                 rescue_margin=1.0, btex_quantile=0.75):
        super(ComputeLoss, self).__init__()
        device = next(model.parameters()).device  # get model device
        h = model.hyp  # hyperparameters

        # HESOD selector-loss isolation:
        #   upstream: released-code behavior, per-pixel weighted BCE only.
        #   paper:    paper-text behavior, focal:dice = 20:1 (Eq. 4 discussion).
        #   coverage: HESOD object-level soft coverage + positive BCE weighting.
        # The paper-text variant is explicit because the released ESOD code
        # leaves its focal/dice lines commented out; these are two distinct
        # reproduction targets rather than interchangeable baselines.
        if selector_loss not in ('upstream', 'paper', 'coverage'):
            raise ValueError(
                f"selector_loss must be 'upstream', 'paper', or 'coverage', got {selector_loss!r}"
            )
        self.selector_loss = selector_loss
        self.lambda_cov = lambda_cov if selector_loss == 'coverage' else 0.0
        self.mask_pos_weight = pos_weight if selector_loss == 'coverage' else None

        # HESOD box-regression ablation switch (HESOD-Experiment-Plan.md SS3.2).
        # 'upstream' preserves the released per-anchor (1-CIoU).mean().
        # 'size_weighted' is the older inverse-area weighting arm. 'sabl' is
        # SSABNet's exact internal CIoU rectification with fixed paper constants
        # kappa=32 px, beta=6, C=12; it changes lbox only.
        if box_loss not in ('upstream', 'size_weighted', 'sabl'):
            raise ValueError(
                f"box_loss must be 'upstream', 'size_weighted', or 'sabl', got {box_loss!r}"
            )
        self.box_loss = box_loss
        self.box_weight_ref_area = box_weight_ref_area
        self.box_weight_max = box_weight_max

        # Define criteria
        BCEcls = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['cls_pw']], device=device))
        BCEobj = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['obj_pw']], device=device))

        # Class label smoothing https://arxiv.org/pdf/1902.04103.pdf eqn 3
        self.cp, self.cn = smooth_BCE(eps=h.get('label_smoothing', 0.0))  # positive, negative BCE targets

        # Focal loss
        g = h['fl_gamma']  # focal loss gamma
        if g > 0:
            BCEcls = FocalLoss(BCEcls, g)
            # BCEobj = FocalLoss(BCEobj, g)
        # else:
        #     BCEobj = QFocalLoss(BCEobj, gamma=1.5, alpha=0.5)

        det = model.module.model[-1] if is_parallel(model) else model.model[-1]  # Detect() module
        self.balance = {3: [4.0, 1.0, 0.4]}.get(det.nl, [4.0, 1.0, 0.25, 0.06, .02])  # P3-P7
        self.ssi = list(det.stride).index(16) if autobalance else 0  # stride 16 index
        self.BCEcls, self.BCEobj, self.gr, self.hyp, self.autobalance = BCEcls, BCEobj, model.gr, h, autobalance
        for k in 'na', 'nc', 'nl', 'anchors', 'anchor_grid', 'stride':
            setattr(self, k, getattr(det, k))
        self.neg_anchor_iou_thres = 0.7
        self.pos_anchor_iou_thres = 0.15
        self.pos_anchor_num = 4
        self.lpixl_critreia = None

        # F6 rescue-ranking + conditional-gate regularization (HESOD-Agri-
        # Proposal.md SS4.2.2). lambda_rescue=lambda_cond=0.0 (default) keeps
        # every other arm (F0/F1/F5/etc.) byte-identical -- need_gate_extras
        # in the forward call is gated on these two being nonzero (train.py),
        # so this whole code path never executes unless explicitly enabled.
        if lambda_rescue < 0 or lambda_cond < 0:
            raise ValueError(f'lambda_rescue and lambda_cond must be >= 0, got {lambda_rescue!r}, {lambda_cond!r}')
        if not (0.0 < tau_low < 1.0):
            raise ValueError(f'tau_low must be in (0,1), got {tau_low!r}')
        self.lambda_rescue, self.lambda_cond = lambda_rescue, lambda_cond
        self.tau_low = tau_low
        # tau_high gates C_sem = "already confidently correct" positives (SS4.2.2);
        # defaults to a symmetric confidence band around the 0.5 decision boundary
        # since the design doc only pre-registers tau_low, not a separate tau_high.
        self.tau_high = tau_high if tau_high is not None else (1.0 - tau_low)
        if not (0.0 < self.tau_high < 1.0):
            raise ValueError(f'tau_high must be in (0,1), got {self.tau_high!r}')
        if rescue_margin < 0:
            raise ValueError(f'rescue_margin must be >= 0, got {rescue_margin!r}')
        if not (0.0 < btex_quantile < 1.0):
            raise ValueError(f'btex_quantile must be in (0,1), got {btex_quantile!r}')
        self.rescue_margin, self.btex_quantile = rescue_margin, btex_quantile
        self._rescue_max_pairs, self._btex_min_bg_cells = 2048, 16
        self.texture_filter = FixedTextureFilter().to(device)
        self.last_lrescue = self.last_lcond = torch.zeros(1, device=device)
        self.last_b_tex_frac = self.last_c_sem_frac = torch.zeros(1, device=device)

    def __call__(self, p, targets, imgsz=None, masks=None, m_weights=None, imgs=None):  # predictions, targets, model
        if len(p) == 3:
            p_det, p_seg, gate_extras = p
        else:
            p_det, p_seg = p
            gate_extras = None
        offsets = []
        device = targets.device
        lcls, lbox, lobj = torch.zeros(1, device=device), torch.zeros(1, device=device), torch.zeros(1, device=device)
        lpixl, larea, ldist = torch.zeros(1, device=device), torch.zeros(1, device=device), torch.zeros(1, device=device)
        lrescue, lcond = torch.zeros(1, device=device), torch.zeros(1, device=device)
        
        if p_det is not None and p_det[0] is not None and p_det[1] is not None:  # stupid
            # ta = time_synchronized()
            if isinstance(p_det, tuple):
                p, offsets = p_det
                tcls, tbox, indices, anchors = self.build_patch_targets(offsets, targets, imgsz)  # targets
            else:
                p = p_det
                tcls, tbox, indices, anchors = self.build_targets(p, targets)
            # print(f'build_targets: {time_synchronized() - ta:.3f}s.')

            # Losses
            for i, pi in enumerate(p):  # layer index, layer predictions
                b, a, gj, gi = indices[i]  # image, anchor, gridy, gridx
                tobj = torch.zeros_like(pi[..., 0], device=device)  # target obj
    
                n = b.shape[0]  # number of targets
                if n:
                    ps = pi[b, a, gj, gi]  # prediction subset corresponding to targets
    
                    # Regression
                    pxy = ps[:, :2].sigmoid() * 2. - 0.5
                    pwh = (ps[:, 2:4].sigmoid() * 2) ** 2 * anchors[i]
                    pbox = torch.cat((pxy, pwh), 1)  # predicted box
                    iou = bbox_iou(pbox.T, tbox[i], x1y1x2y2=False, CIoU=True)  # CIoU for objectness quality
                    if self.box_loss == 'size_weighted':
                        area_cells = (tbox[i][:, 2] * tbox[i][:, 3]).clamp(min=1e-3)
                        box_weight = (self.box_weight_ref_area / area_cells).clamp(1.0, self.box_weight_max)
                        lbox += ((1.0 - iou) * box_weight).mean()  # size-weighted iou loss
                    elif self.box_loss == 'sabl':
                        lbox += sabl_loss(pbox, tbox[i], self.stride[i]).mean()
                    else:
                        lbox += (1.0 - iou).mean()  # iou loss
    
                    # Objectness
                    tobj[b, a, gj, gi] = (1.0 - self.gr) + self.gr * iou.detach().clamp(0).type(tobj.dtype)  # iou ratio
    
                    # Classification
                    if self.nc > 1:  # cls loss (only if multiple classes)
                        t = torch.full_like(ps[:, 5:], self.cn, device=device)  # targets
                        t[range(n), tcls[i]] = self.cp
                        lcls += self.BCEcls(ps[:, 5:], t)  # BCE
    
                    # Append targets to text file
                    # with open('targets.txt', 'a') as file:
                    #     [file.write('%11.5g ' * 4 % tuple(x) + '\n') for x in torch.cat((txy[i], twh[i]), 1)]
    
                obji = self.BCEobj(pi[..., 4].clamp_(-9.21, 9.21), tobj)
                lobj += obji * self.balance[i]  # obj loss
                if self.autobalance:
                    self.balance[i] = self.balance[i] * 0.9999 + 0.0001 / obji.detach().item()
        
        # bs = tobj.shape[0]  # batch size
        bs = p_seg[0].shape[0] if p_seg is not None else tobj.shape[0]
        if self.autobalance:
            self.balance = [x / self.balance[self.ssi] for x in self.balance]
            
        lbox *= self.hyp['box']
        lobj *= self.hyp['obj'] * 0.5 #(0.5 if (len(offsets) and len(offsets[0]) > bs) else 1.)   # adaoff: 0.178
        lcls *= self.hyp['cls']
        
        if masks is not None and p_seg is not None:
            assert len(p_seg) == 1
            lpixl, larea, ldist = self.compute_loss_seg(p_seg[0], masks, targets, weight=m_weights)

            if (self.lambda_rescue > 0 or self.lambda_cond > 0) and gate_extras is not None:
                # gate_extras is None whenever this call didn't request it --
                # notably train.py's own per-epoch validation pass (test.test()
                # -> test.py's model(...) call, which never sets
                # need_gate_extras=True) reuses this same ComputeLoss instance
                # but only ever reads loss_items[:6] (box..dist), never index 6
                # (the "total" that would include lrescue/lcond) -- so skipping
                # here is correct, not a silently-missing training signal.
                #
                # += (not reassignment): compute_gate_losses' pieces are 0-dim
                # .mean() results, matching every other loss term in this
                # method -- += against the pre-initialized shape-(1,) zeros
                # broadcasts correctly; a direct reassignment would leave
                # lrescue/lcond 0-dim and break the final torch.cat below.
                lr, lc = self.compute_gate_losses(p_seg[0], gate_extras[0], targets, imgs)
                lrescue += lr
                lcond += lc

        # lrescue/lcond already ARE the intended weights (lambda_rescue/lambda_cond),
        # applied here directly -- NOT folded into the *0.2 selector-loss group above.
        loss = (lbox + lobj + lcls) * 1.0 + (lpixl + larea + ldist) * 0.2 \
            + lrescue * self.lambda_rescue + lcond * self.lambda_cond
        loss_items = torch.cat((lbox, lobj, lcls, lpixl, larea, ldist, loss)).detach()
        return loss * bs, loss_items

    def build_targets(self, p, targets):
        # Build targets for compute_loss(), input targets(image,class,x,y,w,h), 0~1
        na, nt = self.na, targets.shape[0]  # number of anchors, targets
        tcls, tbox, indices, anch = [], [], [], []
        gain = torch.ones(7, device=targets.device)  # normalized to gridspace gain
        ai = torch.arange(na, device=targets.device).float().view(na, 1).repeat(1, nt)  # same as .repeat_interleave(nt)
        targets = torch.cat((targets.repeat(na, 1, 1), ai[:, :, None]), 2)  # append anchor indices, shape(na,nt,7)

        g = 0.5  # bias
        off = torch.tensor([[0, 0],
                            [1, 0], [0, 1], [-1, 0], [0, -1],  # j,k,l,m
                            # [1, 1], [1, -1], [-1, 1], [-1, -1],  # jk,jm,lk,lm
                            ], device=targets.device).float() * g  # offsets

        for i in range(self.nl):
            anchors = self.anchors[i]
            gain[2:6] = torch.tensor(p[i].shape)[[3, 2, 3, 2]]  # xyxy gain

            # Match targets to anchors
            t = targets * gain
            if nt:
                # Matches
                r = t[:, :, 4:6] / anchors[:, None]  # wh ratio
                j = torch.max(r, 1. / r).max(2)[0] < self.hyp['anchor_t']  # compare
                # j = wh_iou(anchors, t[:, 4:6]) > model.hyp['iou_t']  # iou(3,n)=wh_iou(anchors(3,2), gwh(n,2))
                t = t[j]  # filter shape(nt_,7), [bi, ci, xc, yc, w, h, ai]

                # Offsets
                gxy = t[:, 2:4]  # grid xy
                gxi = gain[[2, 3]] - gxy  # inverse
                j, k = ((gxy % 1. < g) & (gxy > 1.)).T
                l, m = ((gxi % 1. < g) & (gxi > 1.)).T
                j = torch.stack((torch.ones_like(j), j, k, l, m))
                t = t.repeat((5, 1, 1))[j]
                offsets = (torch.zeros_like(gxy)[None] + off[:, None])[j]
            else:
                t = targets[0]
                offsets = 0

            # Define
            b, c = t[:, :2].long().T  # image, class
            gxy = t[:, 2:4]  # grid xy
            gwh = t[:, 4:6]  # grid wh
            gij = (gxy - offsets).long()
            gi, gj = gij.T  # grid xy indices

            # Append
            a = t[:, 6].long()  # anchor indices
            # int(...) not gain[3]-1 directly: gain is float32 (see gain[2:6]
            # assignment above), and newer PyTorch's Tensor.clamp_ raises
            # "result type Float can't be cast to the desired output type
            # long int" when an in-place clamp on a long tensor is given a
            # float-tensor bound -- this line was never exercised by any
            # arm before A0 (every prior arm routes through
            # build_patch_targets, not build_targets, for its sparse/patched
            # detection head), so the dtype mismatch was previously latent,
            # not something this session's changes introduced.
            indices.append((b, a, gj.clamp_(0, int(gain[3]) - 1), gi.clamp_(0, int(gain[2]) - 1)))  # image, anchor, grid indices
            tbox.append(torch.cat((gxy - gij, gwh), 1))  # box
            anch.append(anchors[a])  # anchors
            tcls.append(c)  # class

        return tcls, tbox, indices, anch
  
    def build_patch_targets(self, patch_offsets, targets, imgsz):  # for fast-mode, fixed patch division
        # Build targets for compute_loss(), input targets(image,class,x,y,w,h)
        na, nt = self.na, targets.shape[0]  # number of anchors, targets
        dtype, device = targets.dtype, targets.device
        tcls, tbox, indices, anch = [], [], [], []
        bs, _, height, width = imgsz
        
        gain = torch.ones(7, device=device)  # normalized to gridspace gain
        ai = torch.arange(na, device=device).float().view(na, 1).repeat(1, nt)  # same as .repeat_interleave(nt)
        targets = torch.cat((targets.repeat(na, 1, 1), ai[:, :, None]), 2)  # append anchor indices, shape(na,nt,7)
        bi_ = torch.arange(patch_offsets[0].shape[0], device=device)

        g = 0.5  # bias
        off = torch.tensor([[0, 0],
                            [1, 0], [0, 1], [-1, 0], [0, -1],  # j,k,l,m
                            # [1, 1], [1, -1], [-1, 1], [-1, -1],  # jk,jm,lk,lm
                            ], device=device).float() * g  # offsets

        for i in range(self.nl):
            patch_off = patch_offsets[i]
            anchors = self.anchors[i]
            r = (2 ** (i - 1)) if self.nl == 4 else 2 ** i
            gain[2:6] = torch.tensor([width, height, width, height], dtype=dtype) / (8 * r)  # TODO: from 4 to 32
            # grid_w, grid_h = patch_off[0, [3, 4]] - patch_off[0, [1, 2]]
            grid_wh = patch_off[:1, [3, 4]] - patch_off[:1, [1, 2]]

            # Match targets to anchors
            t = targets * gain
            if nt:
                # Matches
                r = t[:, :, 4:6] / anchors[:, None]  # wh ratio
                j = torch.max(r, 1. / r).max(2)[0] < self.hyp['anchor_t']  # compare
                # j = wh_iou(anchors, t[:, 4:6]) > model.hyp['iou_t']  # iou(3,n)=wh_iou(anchors(3,2), gwh(n,2))
                t = t[j]  # filter, shape(nt_, 7)

                tb, txc, tyc = t[:, [0, 2, 3]].chunk(3, dim=1)  # shape(n,1)
                pb, px1, py1, px2, py2 = (patch_off.T).chunk(5, dim=0)  # shape(1,m)
                contained = (tb == pb) & (txc > px1 - g) & (txc < px2 - g) & (tyc > py1 - g) & (tyc < py2 - g)  # shape(n,m)
                ti, pj = torch.nonzero(contained).T  # i-th target is contained within j-th patch
                t = t[ti]  # shape(n,7)
                
                # Offsets
                gxy = t[:, 2:4]  # grid xy
                gxi = grid_wh - gxy  # inverse
                j, k = ((gxy - gxy.floor() < g) & (gxy > 0.-g)).T
                l, m = ((gxi - gxi.floor() < g) & (gxi > 1.-g)).T
                # j, k = ((gxy % 1. < g) & (gxy > 1.)).T
                # l, m = ((gxi % 1. < g) & (gxi > 1.)).T
                j = torch.stack((torch.ones_like(j), j, k, l, m))
                
                t[:, 0] = bi_[pj]  # converted batch-indices
                t[:, 2:4] -= patch_off[pj, 1:3]  # converted xc, yc (minus px1, py1)

                t = t.repeat((5, 1, 1))[j]
                offsets = (torch.zeros_like(gxy)[None] + off[:, None])[j]

            else:
                t = targets[0]
                offsets = 0

            # Define
            b, c = t[:, :2].long().T  # image, class
            gxy = t[:, 2:4]  # grid xy
            gwh = t[:, 4:6]  # grid wh
            gij = (gxy - offsets).long()
            gi, gj = gij.T  # grid xy indices

            # Append
            a = t[:, 6].long()  # anchor indices
            # assert ((gj >= 0) & (gj <= grid_wh[0,1] - 1) & (gi >= 0) & (gi <= grid_wh[0,0] - 1)).all()
            # indices.append((b, a, gj.clamp_(0, grid_wh[0,1] - 1), gi.clamp_(0, grid_wh[0,0] - 1)))  # image, anchor, grid indices
            indices.append((b, a, gj, gi))  # image, anchor, grid indices
            tbox.append(torch.cat((gxy - gij, gwh), 1))  # box
            anch.append(anchors[a])  # anchors
            tcls.append(c)  # class

        return tcls, tbox, indices, anch

    def compute_loss_seg(self, p, masks, targets, weight=None):
        dtype, device = targets.dtype, targets.device
        bs, nc, ny, nx = masks.shape
        assert nc == 1
        lpixl, larea, ldist = torch.zeros(1, device=device), torch.zeros(1, device=device), \
                              torch.zeros(1, device=device)

        if self.selector_loss == 'paper':
            # ESOD paper: focal loss and dice loss with a 20:1 ratio. This is
            # deliberately separate from the released-code weighted BCE path.
            ldist += self.sigmoid_focal_loss(p, masks) * 20.0
            larea += self.dice_loss(p, masks)
        else:
            pos_weight = None
            if self.selector_loss == 'coverage' and self.mask_pos_weight is not None:
                pos_weight = torch.as_tensor(self.mask_pos_weight, device=device, dtype=p.dtype)
            lpixl += F.binary_cross_entropy_with_logits(p, masks, weight=weight, pos_weight=pos_weight)

        nt = targets.shape[0]
        if self.selector_loss == 'coverage' and nt:
            larea += self.compute_coverage_loss(p, targets, ny, nx) * self.lambda_cov

        return lpixl, larea, ldist

    def compute_coverage_loss(self, p, targets, ny, nx):
        """Object-level soft coverage loss (HESOD-Proposal.md SS3.3, SS5.4).

        p: (bs, 1, ny, nx) raw selector logits.
        targets: (nt, 6) [image_idx, class, xc, yc, w, h], normalized [0, 1].

        For each GT object j, N(j) is the set of selector cells whose grid
        footprint overlaps j's box. The soft probability that j is covered by
        at least one retained cell is p_cover = 1 - prod_{i in N(j)}(1 - s_i),
        s_i = sigmoid(p_i). Loss is -w_j * log(p_cover), w_j larger for
        smaller objects, so it only takes one confident cell per object to
        drive this term to ~0 -- unlike per-pixel BCE, it does not push every
        cell touching an object toward 1.
        """
        device = p.device
        s = p[:, 0].sigmoid()  # (bs, ny, nx)
        eps = 1e-6

        # HESOD: reference object area (in grid cells) below which the tiny-size
        # weight saturates, and the weight cap itself. Kept as fixed constants for
        # this MVP rather than exposed flags -- see HESOD-Proposal.md SS3.4 on
        # lambda/weight tuning being an open research knob, not a solved default.
        ref_area_cells = 4.0
        max_weight = 5.0

        losses = []
        for bi in range(s.shape[0]):
            obj = targets[targets[:, 0] == bi]
            if obj.shape[0] == 0:
                continue
            xc, yc = obj[:, 2] * nx, obj[:, 3] * ny
            w, h = (obj[:, 4] * nx).clamp(min=1e-3), (obj[:, 5] * ny).clamp(min=1e-3)
            x1 = (xc - w / 2).floor().clamp(0, nx - 1).long()
            y1 = (yc - h / 2).floor().clamp(0, ny - 1).long()
            x2 = torch.maximum((xc + w / 2).ceil().clamp(0, nx).long(), x1 + 1)
            y2 = torch.maximum((yc + h / 2).ceil().clamp(0, ny).long(), y1 + 1)

            area_cells = (w * h).clamp(min=1e-3)
            weight = (ref_area_cells / area_cells).clamp(1.0, max_weight)

            for j in range(obj.shape[0]):
                region = s[bi, y1[j]:y2[j], x1[j]:x2[j]]
                p_cover = 1.0 - torch.prod(1.0 - region)
                losses.append(-weight[j] * torch.log(p_cover.clamp(min=eps)))

        if not losses:
            return torch.zeros(1, device=device)
        return torch.stack(losses).mean()

    def _positive_cell_mask(self, targets, bs, ny, nx):
        """Binary y_i grid (bs,ny,nx): True where a GT box footprint covers
        cell (row,col). Reuses compute_coverage_loss's exact floor/ceil
        box-to-cell arithmetic (SS6.2.1) so the rescue-ranking/conditional-
        gate-reg losses' y_i=1 set matches the coverage loss's own notion of
        "covered by this object", not a separately-invented definition.
        """
        device = targets.device
        y = torch.zeros(bs, ny, nx, dtype=torch.bool, device=device)
        for bi in range(bs):
            obj = targets[targets[:, 0] == bi]
            if obj.shape[0] == 0:
                continue
            xc, yc = obj[:, 2] * nx, obj[:, 3] * ny
            w, h = (obj[:, 4] * nx).clamp(min=1e-3), (obj[:, 5] * ny).clamp(min=1e-3)
            x1 = (xc - w / 2).floor().clamp(0, nx - 1).long()
            y1 = (yc - h / 2).floor().clamp(0, ny - 1).long()
            x2 = torch.maximum((xc + w / 2).ceil().clamp(0, nx).long(), x1 + 1)
            y2 = torch.maximum((yc + h / 2).ceil().clamp(0, ny).long(), y1 + 1)
            for j in range(obj.shape[0]):
                y[bi, y1[j]:y2[j], x1[j]:x2[j]] = True
        return y

    def _texture_hard_negative_mask(self, imgs, bg_mask, ny, nx):
        """B_tex (bs,ny,nx): background cells (bg_mask, i.e. ~y_i) whose
        texture score is at/above this batch's btex_quantile of the
        background-cell score distribution. `imgs` is the raw model-input
        tensor (bs,3,H,W) -- content-only, independent of every learned
        quantity (see FixedTextureFilter's docstring).
        """
        with torch.no_grad():
            gray = imgs.float().mean(dim=1, keepdim=True)
            stride = imgs.shape[-1] // nx
            texture = self.texture_filter(gray)
            texture_score = F.avg_pool2d(texture, kernel_size=stride, stride=stride)[:, 0]
            # avg_pool2d's output grid can be off by a cell from (ny,nx) if
            # H/W isn't an exact multiple of stride; crop defensively rather
            # than assume exact divisibility.
            texture_score = texture_score[:, :ny, :nx]
            bg_scores = texture_score[bg_mask]
            if bg_scores.numel() < self._btex_min_bg_cells:
                return torch.zeros_like(bg_mask)
            thresh = torch.quantile(bg_scores, self.btex_quantile)
            return bg_mask & (texture_score >= thresh)

    def compute_rescue_loss(self, u, q, y_mask, b_tex):
        """L_rescue (HESOD-Agri-Proposal.md SS4.2.2): pairs real GT-positive,
        low-semantic-confidence cells (P_rescue's i) against texture
        hard-negative cells (P_rescue's j) within the same image, and trains
        the gate so real rescued targets outrank texture background:
        L = mean over (i,j) in P_rescue of softplus(margin - u_i + u_j).
        Mining is per-image (pairs never cross images); pooled into one flat
        mean across the batch (a single P_rescue set, matching the design
        doc's literal wording, not a mean of per-image means).
        `self._rescue_max_pairs` caps the O(n_i * n_j) pair count per image --
        a fixed internal constant for this pre-registered pilot, not a swept
        hyperparameter.
        """
        device = u.device
        losses = []
        for bi in range(u.shape[0]):
            i_idx = (y_mask[bi] & (q[bi] < self.tau_low)).nonzero(as_tuple=False)
            j_idx = b_tex[bi].nonzero(as_tuple=False)
            n_i, n_j = i_idx.shape[0], j_idx.shape[0]
            if n_i == 0 or n_j == 0:
                continue
            ii = i_idx.repeat_interleave(n_j, dim=0)
            jj = j_idx.repeat(n_i, 1)
            if ii.shape[0] > self._rescue_max_pairs:
                sel = torch.randperm(ii.shape[0], device=device)[:self._rescue_max_pairs]
                ii, jj = ii[sel], jj[sel]
            u_i = u[bi][ii[:, 0], ii[:, 1]]
            u_j = u[bi][jj[:, 0], jj[:, 1]]
            losses.append(F.softplus(self.rescue_margin - u_i + u_j))
        if not losses:
            return torch.zeros(1, device=device)
        return torch.cat(losses).mean()

    def compute_cond_loss(self, a, c_sem_mask, b_tex_mask):
        """L_cond = mean_{C_sem}(a_i) + mean_{B_tex}(a_i) -- two separate
        means summed, not a mean over their union (matches the design doc
        formula literally; C_sem subset of {y_i=1}, B_tex subset of {y_i=0}
        are disjoint by construction, so this never double-counts a cell)."""
        device = a.device
        terms = []
        if c_sem_mask.any():
            terms.append(a[c_sem_mask].mean())
        if b_tex_mask.any():
            terms.append(a[b_tex_mask].mean())
        if not terms:
            return torch.zeros(1, device=device)
        return sum(terms)

    def compute_gate_losses(self, p_fused, extras, targets, imgs):
        """Assembles P_rescue/C_sem/B_tex for this batch and returns
        (lrescue, lcond), each 0-dim, meant to be accumulated via `+=`
        against a pre-initialized shape-(1,) zero tensor by the caller (see
        __call__) -- matching every other loss term in this class.
        """
        if imgs is None:
            raise RuntimeError('compute_gate_losses requires imgs (raw model input) to build B_tex')
        bs, _, ny, nx = p_fused.shape
        u, q, a = p_fused[:, 0], extras['q'][:, 0], extras['a'][:, 0]

        y_mask = self._positive_cell_mask(targets, bs, ny, nx)
        b_tex = self._texture_hard_negative_mask(imgs, ~y_mask, ny, nx)
        c_sem = y_mask & (q > self.tau_high)

        lrescue = self.compute_rescue_loss(u, q, y_mask, b_tex) if self.lambda_rescue > 0 \
            else torch.zeros(1, device=u.device)
        lcond = self.compute_cond_loss(a, c_sem, b_tex) if self.lambda_cond > 0 \
            else torch.zeros(1, device=u.device)

        self.last_lrescue, self.last_lcond = lrescue.detach(), lcond.detach()
        self.last_b_tex_frac = b_tex.float().mean().detach()
        self.last_c_sem_frac = c_sem.float().mean().detach()
        return lrescue, lcond

    @staticmethod
    def dice_loss(inputs, targets):
        """
        Compute the DICE loss, similar to generalized IOU for masks
        Args:
            inputs: A float tensor of arbitrary shape.
                    The predictions for each example.
            targets: A float tensor with the same shape as inputs. Stores the binary
                    classification label for each element in inputs
                    (0 for the negative class and 1 for the positive class).
        """
        inputs = inputs.sigmoid().flatten(1)
        targets = targets.flatten(1)
        numerator = 2 * (inputs * targets).sum(-1)
        denominator = inputs.sum(-1) + targets.sum(-1)
        loss = 1 - (numerator + 1) / (denominator + 1)
        return loss.mean()

    @staticmethod
    def sigmoid_focal_loss(inputs, targets, alpha: float = 0.25, gamma: float = 2):
        """
        Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
        Args:
            inputs: A float tensor of arbitrary shape.
                    The predictions for each example.
            targets: A float tensor with the same shape as inputs. Stores the binary
                    classification label for each element in inputs
                    (0 for the negative class and 1 for the positive class).
            alpha: (optional) Weighting factor in range (0,1) to balance
                    positive vs negative examples. Default = -1 (no weighting).
            gamma: Exponent of the modulating factor (1 - p_t) to
                balance easy vs hard examples.
        Returns:
            Loss tensor
        """
        prob = inputs.sigmoid()
        ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
        p_t = prob * targets + (1 - prob) * (1 - targets)
        loss = ce_loss * ((1 - p_t) ** gamma)

        if alpha >= 0:
            alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
            loss = alpha_t * loss

        return loss.mean()

    @staticmethod
    def quality_dice_loss(inputs, targets, weight=None, gamma: float = 2):
        """
        Compute the DICE loss, similar to generalized IOU for masks
        Args:
            inputs: A float tensor of arbitrary shape.
                    The predictions for each example.
            targets: A float tensor with the same shape as inputs. Stores the binary
                    classification label for each element in inputs
                    (0 for the negative class and 1 for the positive class).
        """
        inputs = inputs.sigmoid().flatten(1)
        targets = targets.flatten(1)
        if weight is not None:
            weight = weight.flatten(1)
            inputs = inputs * weight
            targets = targets * weight

        numerator = 2 * (inputs - targets).abs().sum(-1)
        denominator = inputs.sum(-1) + targets.sum(-1)
        loss = (numerator + 1) / (denominator + 1)
        return loss.mean()

    @staticmethod
    def sigmoid_quality_focal_loss(inputs, targets, weight=None, alpha: float = 0.25, gamma: float = 2):
        """
        Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
        Args:
            inputs: A float tensor of arbitrary shape.
                    The predictions for each example.
            targets: A float tensor with the same shape as inputs. Stores the binary
                    classification label for each element in inputs
                    (0 for the negative class and 1 for the positive class).
            alpha: (optional) Weighting factor in range (0,1) to balance
                    positive vs negative examples. Default = -1 (no weighting).
            gamma: Exponent of the modulating factor (1 - p_t) to
                balance easy vs hard examples.
        Returns:
            Loss tensor
        """
        prob = inputs.sigmoid()
        ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, weight=weight, reduction="none")
        loss = ce_loss * ((prob - targets).abs() ** gamma)

        if alpha >= 0:
            alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
            loss = alpha_t * loss

        return loss.mean()
