import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math
import random


def get_loss_fn(loss_fn_name, smoothing=0.05):

    loss_fn_dict = {
        'crossentropy': nn.CrossEntropyLoss(),
        'focalloss': FocalLoss(),
        'focalcosineloss': FocalCosineLoss(smoothing=smoothing),
        'labelsmoothing': MyLabelSmoothingLoss(smoothing=smoothing),
        'bitemperedloss': BiTemperedLoss(t1=0.5, t2=1.0, smoothing=smoothing)
    }

    return loss_fn_dict[loss_fn_name]

class FocalLoss(nn.Module):
    def __init__(self, gamma=0, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha,(float, int)): self.alpha = torch.Tensor([alpha,1-alpha])
        if isinstance(alpha, list): self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim()>2:
            input = input.view(input.size(0),input.size(1),-1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1,2)    # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1,input.size(2))   # N,H*W,C => N*H*W,C
        target = target.view(-1,1)

        logpt = F.log_softmax(input)
        logpt = logpt.gather(1,target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type()!=input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0,target.data.view(-1))
            logpt = logpt * Variable(at)

        loss = -1 * (1-pt)**self.gamma * logpt
        if self.size_average: return loss.mean()
        else: return loss.sum()


# Reference https://www.kaggle.com/c/cassava-leaf-disease-classification/discussion/203271
class FocalCosineLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, xent=.1, classes=5, smoothing=0.0, p=0.5):
        super(FocalCosineLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.xent = xent
        self.classes = classes
        self.smoothing = smoothing
        self.p = p

        self.y = torch.Tensor([1]).cuda()

    def forward(self, input, target, reduction="mean"):
        t = F.one_hot(target, num_classes=self.classes)
        if self.smoothing > 0 and random.random() > self.p:
            t = (t - self.smoothing).abs()

        cosine_loss = F.cosine_embedding_loss(input, t, self.y, reduction=reduction)

        cent_loss = F.cross_entropy(F.normalize(input), target, reduce=False)
        pt = torch.exp(-cent_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * cent_loss

        if reduction == "mean":
            focal_loss = torch.mean(focal_loss)

        return cosine_loss + self.xent * focal_loss



# ====================================================
# Label Smoothing
# Reference https://www.kaggle.com/piantic/train-cassava-starter-using-label-smoothing
# ====================================================
class LabelSmoothingLoss(nn.Module):
    def __init__(self, classes=5, smoothing=0.0, dim=-1):
        super(LabelSmoothingLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.cls = classes
        self.dim = dim

    def forward(self, pred, target):
        pred = pred.log_softmax(dim=self.dim)
        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.cls - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))



class MyLabelSmoothingLoss(nn.Module):
    def __init__(self, classes=5, smoothing=0.0, p=0.5):
        super(MyLabelSmoothingLoss, self).__init__()
        self.classes = classes
        self.smoothing = smoothing
        self.p = p

    def forward(self, pred, target):
        pred = torch.softmax(pred, dim=1)
        _target = torch.eye(self.classes)[target].squeeze()
        _target = _target.cuda()

        if random.random() > self.p:
            _target = (_target - self.smoothing).abs()

        loss = F.binary_cross_entropy_with_logits(pred, _target)

        return loss



# Bi-Tempered-loss
# reference https://github.com/mlpanda/bi-tempered-loss-pytorch/blob/master/bi_tempered_loss.py

def log_t(u, t):
    """Compute log_t for `u`."""

    if t == 1.0:
        return torch.log(u)
    else:
        return (u ** (1.0 - t) - 1.0) / (1.0 - t)


def exp_t(u, t):
    """Compute exp_t for `u`."""

    if t == 1.0:
        return torch.exp(u)
    else:
        return torch.relu(1.0 + (1.0 - t) * u) ** (1.0 / (1.0 - t))


def compute_normalization_fixed_point(activations, t, num_iters=5):
    """Returns the normalization value for each example (t > 1.0).
    Args:
    activations: A multi-dimensional tensor with last dimension `num_classes`.
    t: Temperature 2 (> 1.0 for tail heaviness).
    num_iters: Number of iterations to run the method.
    Return: A tensor of same rank as activation with the last dimension being 1.
    """

    mu = torch.max(activations, dim=-1).values.view(-1, 1)
    normalized_activations_step_0 = activations - mu

    normalized_activations = normalized_activations_step_0
    i = 0
    while i < num_iters:
        i += 1
        logt_partition = torch.sum(exp_t(normalized_activations, t), dim=-1).view(-1, 1)
        normalized_activations = normalized_activations_step_0 * (logt_partition ** (1.0 - t))

    logt_partition = torch.sum(exp_t(normalized_activations, t), dim=-1).view(-1, 1)

    return -log_t(1.0 / logt_partition, t) + mu


def compute_normalization(activations, t, num_iters=5):
    """Returns the normalization value for each example.
    Args:
    activations: A multi-dimensional tensor with last dimension `num_classes`.
    t: Temperature 2 (< 1.0 for finite support, > 1.0 for tail heaviness).
    num_iters: Number of iterations to run the method.
    Return: A tensor of same rank as activation with the last dimension being 1.
    """

    if t < 1.0:
        return None # not implemented as these values do not occur in the authors experiments...
    else:
        return compute_normalization_fixed_point(activations, t, num_iters)


def tempered_softmax(activations, t, num_iters=5):
    """Tempered softmax function.
    Args:
    activations: A multi-dimensional tensor with last dimension `num_classes`.
    t: Temperature tensor > 0.0.
    num_iters: Number of iterations to run the method.
    Returns:
    A probabilities tensor.
    """

    if t == 1.0:
        normalization_constants = torch.log(torch.sum(torch.exp(activations), dim=-1))
    else:
        normalization_constants = compute_normalization(activations, t, num_iters)

    return exp_t(activations - normalization_constants, t)


def bi_tempered_logistic_loss(activations, labels, t1, t2, label_smoothing=0.0, num_iters=5):

    """Bi-Tempered Logistic Loss with custom gradient.
    Args:
    activations: A multi-dimensional tensor with last dimension `num_classes`.
    labels: A tensor with shape and dtype as activations.
    t1: Temperature 1 (< 1.0 for boundedness).
    t2: Temperature 2 (> 1.0 for tail heaviness, < 1.0 for finite support).
    label_smoothing: Label smoothing parameter between [0, 1).
    num_iters: Number of iterations to run the method.
    Returns:
    A loss tensor.
    """

    if label_smoothing > 0.0:
        num_classes = labels.shape[-1]
        labels = (1 - num_classes / (num_classes - 1) * label_smoothing) * labels + label_smoothing / (num_classes - 1)

    probabilities = tempered_softmax(activations, t2, num_iters)

    temp1 = (log_t(labels + 1e-10, t1) - log_t(probabilities, t1)) * labels
    temp2 = (1 / (2 - t1)) * (torch.pow(labels, 2 - t1) - torch.pow(probabilities, 2 - t1))
    loss_values = temp1 - temp2

    return torch.sum(loss_values, dim=-1)


class BiTemperedLoss(nn.Module):
    def __init__(self, t1, t2, smoothing):
        super(BiTemperedLoss, self).__init__()
        self.t1 = t1
        self.t2 = t2
        self.smoothing = smoothing

    def forward(self, pred, label):
        return bi_tempered_logistic_loss(pred, label, self.t1, self.t2, self.smoothing, num_iters=5)
