import torch 
import torch.nn as nn 
import torch.nn.functional as F

class WGAN(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, fake, real=None, mode='D'):
        assert mode in ['D', 'G']
        if mode == 'D':
            assert real  is not None
            return torch.mean(fake) - torch.mean(real)
        else:
            assert real is None 
            return -torch.mean(fake)


class IOULoss(nn.Module):
    def __init__(self, eps = 1e-6,  reduction = 'avg'):
        super().__init__()
        assert reduction in ('avg', 'sum', None), 'Unexpected reduction (avg, sum, None)'
        self.reduction = reduction
        self.eps = eps
        
    def forward(self, logits, mask):
        '''
            pred : (.., Classes, H, W)
            mask : (.., Classes, H, W)
        '''
        pred = torch.sigmoid(logits)
        batch_size = pred.shape[0]
        classes = pred.shape[1]

        intersection = pred * mask
        union = pred + mask - intersection
        
        intersection = intersection.view(batch_size,classes, -1).sum(2)
        union = union.view(batch_size, classes, -1).sum(2)

        loss = (intersection + self.eps) / (union + self.eps)
        loss = 1 - loss

        if self.reduction == 'avg':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()

        return loss


class BinaryDiceLoss(nn.Module):
    def __init__(self, eps = 1e-6, reduction = 'avg', deg = 1):
        super().__init__()
        assert reduction in ('avg', 'sum', None), 'Unexpected reduction'
        self.reduction = reduction
        self.eps = eps
        self.deg = deg

    def forward(self, pred, mask):
        batch_size = pred.shape[0]
        
        pred = pred.view(batch_size, -1)
        mask = mask.view(batch_size, -1)

        intersection = (pred * mask).sum(1) + self.eps 
        addition = (pred ** self.deg + mask ** self.deg).sum(1) + self.eps
        dice = 2 * intersection / addition
        loss = 1 - dice 

        if self.reduction == 'avg':
            return loss.mean()

        if self.reduction == 'sum':
            return loss.sum()

        return loss


class FocalLoss(nn.Module):
    def __init__(self,gamma, alpha = None, reduction = None):
        super().__init__()
        assert reduction in ['avg', 'sum', None], 'Unexpected reduction'
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
        self.criterion = nn.BCEWithLogitsLoss(reduce = None)

    def forward(self, logits, mask):
        '''
            mask: (B, 1, H, W)
            pred: (B, 1, H, W)

        '''
        pred = torch.sigmoid(logits)
        ce = self.criterion(logits, mask)
        alpha = mask * self.alpha + (1. - mask) * (1.0 - self.alpha)
        pt = torch.where(mask == 1, pred, 1 - pred)
        loss =  alpha * (1.0 - pt) ** self.gamma * ce

        if self.reduction == 'avg':
            return loss.mean()

        if self.reduction == 'sum':
            return loss.sum()

        return loss


class TverskyLoss(nn.Module):
    def __init__(self, smooth = 1, alpha = 0.7, reduction = 'avg'):
        super().__init__()

        self.smooth = smooth
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, logits, mask):
        pred = torch.sigmoid(logits)

        batch_size = pred.shape[0]
        pred = pred.contiguous().view(batch_size, -1)
        mask = mask.contiguous().view(batch_size, -1)

        true_positive = (pred * mask).sum(1)
        false_positive = (pred * (1 - mask)).sum(1)
        false_negative = ((1-pred) * mask).sum(1)

        loss = (true_positive + self.smooth) / (true_positive + self.alpha * false_positive + (1 - self.alpha) * false_negative + self.smooth)
        loss = 1 - loss

        if self.reduction == 'avg':
            return loss.mean()

        if self.reduction == 'sum':
            return loss.sum()

        return loss


class FocalTverskyLoss(nn.Module):
    def __init__(self, smooth = 1, alpha = 0.7, gamma = 0.75, reduction = 'avg'):
        super().__init__()
        self.tv = TverskyLoss(smooth = smooth, alpha = alpha, reduction = reduction)
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, pred, mask):
        loss = self.tv(pred, mask)
        return loss ** self.gamma