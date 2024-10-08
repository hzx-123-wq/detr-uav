import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics.utils.metrics import bbox_iou
from ultralytics.utils.checks import check_version
from ultralytics.utils.ops import crop_mask, xywh2xyxy, xyxy2xywh
from ultralytics.utils.tal import TaskAlignedAssigner, dist2bbox, make_anchors

import functools

# ------------------------------BiliBili 魔鬼面具up主------------------------------
class RTDETRLogicLoss(nn.Module):
    def __init__(self, hyp):
        super().__init__()

        self.hyp = hyp
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    
    def power_transform(self, array, power=2):
        return torch.where(array < 0.5, array ** power, array ** (1/power))
    
    def forward(self, s_p, t_p, batch):
        lcls = torch.zeros(1, device=self.device)  # class loss
        lbox_iou = torch.zeros(1, device=self.device)  # box iou loss
        lbox_l1 = torch.zeros(1, device=self.device)  # box l1 loss
        
        s_dec_bboxes, s_dec_scores, s_enc_bboxes, s_enc_scores, s_dn_meta = s_p
        t_dec_bboxes, t_dec_scores, t_enc_bboxes, t_enc_scores, t_dn_meta = t_p[1]
        
        s_dec_bboxes, s_dec_scores = s_dec_bboxes[:, -1], s_dec_scores[:, -1]
        t_dec_bboxes, t_dec_scores = t_dec_bboxes[:, -1], t_dec_scores[:, -1]
        
        if s_dn_meta is not None:
            _, s_dec_bboxes = torch.split(s_dec_bboxes, s_dn_meta['dn_num_split'], dim=1)
            _, s_dec_scores = torch.split(s_dec_scores, s_dn_meta['dn_num_split'], dim=1)
            s_dec_bboxes, s_dec_scores = s_dec_bboxes[-1:], s_dec_scores[-1:]
        # if t_dn_meta is not None:
        #     _, t_dec_bboxes = torch.split(t_dec_bboxes, t_dn_meta['dn_num_split'], dim=2)
        #     _, t_dec_scores = torch.split(t_dec_scores, t_dn_meta['dn_num_split'], dim=2)
        
        t_obj_scale = t_dec_scores.sigmoid().max(-1)[0].unsqueeze(-1)
        
        lbox_l1 = F.l1_loss(s_dec_bboxes, t_dec_bboxes, reduction='none') * t_obj_scale.repeat((1, 1, 4))
        # lbox_l1 = F.l1_loss(s_dec_bboxes, t_dec_bboxes, reduction='none') * self.power_transform(t_obj_scale).repeat((1, 1, 4))
        lbox_iou = (1.0 - bbox_iou(s_dec_bboxes, t_dec_bboxes, xywh=True, GIoU=True)) * self.power_transform(t_obj_scale)
        lcls = nn.BCEWithLogitsLoss(reduction='none')(s_dec_scores, t_dec_scores.sigmoid()).mean(1)
        
        lbox_l1 = lbox_l1.sum() / batch['bboxes'].size(0) * 5
        lbox_iou = lbox_iou.sum() / batch['bboxes'].size(0) * 2
        lcls = lcls.sum() / (batch['bboxes'].size(0) / t_obj_scale.size(1))
        
        return lbox_l1 + lbox_iou + lcls

# ------------------------------BiliBili 魔鬼面具up主------------------------------ 
class FeatureLoss(nn.Module):
    def __init__(self,
                 channels_s,
                 channels_t,
                 distiller='cwd'):
        super(FeatureLoss, self).__init__()

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.align_module = nn.ModuleList([
            nn.Conv2d(channel, tea_channel, kernel_size=1, stride=1,
                      padding=0).to(device) if channel != tea_channel else nn.Identity()
            for channel, tea_channel in zip(channels_s, channels_t)
        ])
        self.norm = [
            nn.BatchNorm2d(tea_channel, affine=False).to(device)
            for tea_channel in channels_t
        ]

        if (distiller == 'mimic'):
            self.feature_loss = MimicLoss(channels_s, channels_t)
        elif (distiller == 'mgd'):
            self.feature_loss = MGDLoss(channels_s, channels_t)
        elif (distiller == 'cwd'):
            self.feature_loss = CWDLoss(channels_s, channels_t)
        elif (distiller == 'chsim'):
            self.feature_loss = ChSimLoss(channels_s, channels_t)
        elif (distiller == 'sp'):
            self.feature_loss = SPKDLoss(channels_s, channels_t)
        else:
            raise NotImplementedError

    def forward(self, y_s, y_t):
        assert len(y_s) == len(y_t)
        tea_feats = []
        stu_feats = []

        for idx, (s, t) in enumerate(zip(y_s, y_t)):
            s = self.align_module[idx](s)
            s = self.norm[idx](s)
            t = self.norm[idx](t)
            tea_feats.append(t)
            stu_feats.append(s)

        loss = self.feature_loss(stu_feats, tea_feats)
        return loss

# ------------------------------BiliBili 魔鬼面具up主------------------------------
class MimicLoss(nn.Module):
    def __init__(self, channels_s, channels_t):
        super(MimicLoss, self).__init__()
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.mse = nn.MSELoss()

    def forward(self, y_s, y_t):
        """Forward computation.
        Args:
            y_s (list): The student model prediction with
                shape (N, C, H, W) in list.
            y_t (list): The teacher model prediction with
                shape (N, C, H, W) in list.
        Return:
            torch.Tensor: The calculated loss value of all stages.
        """
        assert len(y_s) == len(y_t)
        losses = []
        for idx, (s, t) in enumerate(zip(y_s, y_t)):
            assert s.shape == t.shape
            losses.append(self.mse(s, t))
        loss = sum(losses)
        return loss

# ------------------------------BiliBili 魔鬼面具up主------------------------------
class MGDLoss(nn.Module):
    def __init__(self,
                 channels_s,
                 channels_t,
                 alpha_mgd=0.00002,
                 lambda_mgd=0.65):
        super(MGDLoss, self).__init__()
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.alpha_mgd = alpha_mgd
        self.lambda_mgd = lambda_mgd

        self.generation = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(channel, channel, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel, channel, kernel_size=3,
                          padding=1)).to(device) for channel in channels_t
        ])

    def forward(self, y_s, y_t):
        """Forward computation.
        Args:
            y_s (list): The student model prediction with
                shape (N, C, H, W) in list.
            y_t (list): The teacher model prediction with
                shape (N, C, H, W) in list.
        Return:
            torch.Tensor: The calculated loss value of all stages.
        """
        assert len(y_s) == len(y_t)
        losses = []
        for idx, (s, t) in enumerate(zip(y_s, y_t)):
            assert s.shape == t.shape
            losses.append(self.get_dis_loss(s, t, idx) * self.alpha_mgd)
        loss = sum(losses)
        return loss

    def get_dis_loss(self, preds_S, preds_T, idx):
        loss_mse = nn.MSELoss(reduction='sum')
        N, C, H, W = preds_T.shape

        device = preds_S.device
        mat = torch.rand((N, 1, H, W)).to(device)
        mat = torch.where(mat > 1 - self.lambda_mgd, 0, 1).to(device)

        masked_fea = torch.mul(preds_S, mat)
        new_fea = self.generation[idx](masked_fea)

        dis_loss = loss_mse(new_fea, preds_T) / N

        return dis_loss

# ------------------------------BiliBili 魔鬼面具up主------------------------------
class CWDLoss(nn.Module):
    """PyTorch version of `Channel-wise Distillation for Semantic Segmentation.
    <https://arxiv.org/abs/2011.13256>`_.
    """
    def __init__(self, channels_s, channels_t, tau=1.0):
        super(CWDLoss, self).__init__()
        self.tau = tau

    def forward(self, y_s, y_t):
        """Forward computation.
        Args:
            y_s (list): The student model prediction with
                shape (N, C, H, W) in list.
            y_t (list): The teacher model prediction with
                shape (N, C, H, W) in list.
        Return:
            torch.Tensor: The calculated loss value of all stages.
        """
        assert len(y_s) == len(y_t)
        losses = []

        for idx, (s, t) in enumerate(zip(y_s, y_t)):
            assert s.shape == t.shape
            N, C, H, W = s.shape
            # normalize in channel diemension
            softmax_pred_T = F.softmax(t.view(-1, W * H) / self.tau,
                                       dim=1)  # [N*C, H*W]

            logsoftmax = torch.nn.LogSoftmax(dim=1)
            cost = torch.sum(
                softmax_pred_T * logsoftmax(t.view(-1, W * H) / self.tau) -
                softmax_pred_T * logsoftmax(s.view(-1, W * H) / self.tau)) * (
                    self.tau**2)
            # cost = torch.sum(-softmax_pred_T * logsoftmax(s.view(-1, W * H)/self.tau)) * (self.tau ** 2)

            losses.append(cost / (C * N))
        loss = sum(losses)

        return loss

# ------------------------------BiliBili 魔鬼面具up主------------------------------
class ChSimLoss(nn.Module):
    '''
    A loss module for Inter-Channel Correlation for Knowledge Distillation (ICKD).
    https://openaccess.thecvf.com/content/ICCV2021/html/Liu_Exploring_Inter-Channel_Correlation_for_Diversity-Preserved_Knowledge_Distillation_ICCV_2021_paper.html
    '''
    def __init__(self, channels_s, channels_t) -> None:
        super().__init__()
    
    @staticmethod
    def batch_loss(f_s, f_t):
        bsz, ch = f_s.shape[0], f_s.shape[1]
        f_s = f_s.view(bsz, ch, -1)
        f_t = f_t.view(bsz, ch, -1)
        emd_s = torch.bmm(f_s, f_s.permute(0, 2, 1))
        emd_s = torch.nn.functional.normalize(emd_s, dim=2)

        emd_t = torch.bmm(f_t, f_t.permute(0, 2, 1))
        emd_t = torch.nn.functional.normalize(emd_t, dim=2)

        g_diff = emd_s - emd_t
        loss = (g_diff * g_diff).view(bsz, -1).sum() / (ch * bsz * bsz)
        return loss
    
    def forward(self, y_s, y_t):
        assert len(y_s) == len(y_t)
        losses = []
        
        for idx, (s, t) in enumerate(zip(y_s, y_t)):
            assert s.shape == t.shape
            losses.append(self.batch_loss(s, t) / s.size(0))
        loss = sum(losses)
        return loss

# ------------------------------BiliBili 魔鬼面具up主------------------------------
class SPKDLoss(nn.Module):
    '''
    Similarity-Preserving Knowledge Distillation
    https://arxiv.org/pdf/1907.09682.pdf
    '''

    def __init__(self, channels_s, channels_t):
        super(SPKDLoss, self).__init__()

    def matmul_and_normalize(self, z):
        z = torch.flatten(z, 1)
        return F.normalize(torch.matmul(z, torch.t(z)), 1)

    def forward(self, y_s, y_t):
        assert len(y_s) == len(y_t)
        losses = []
        
        for idx, (s, t) in enumerate(zip(y_s, y_t)):
            assert s.shape == t.shape
            g_t = self.matmul_and_normalize(t)
            g_s = self.matmul_and_normalize(s)

            sp_loss = torch.norm(g_t - g_s) ** 2
            sp_loss = sp_loss.sum() / s.size(0)
            losses.append(sp_loss)
        loss = sum(losses)
        return loss