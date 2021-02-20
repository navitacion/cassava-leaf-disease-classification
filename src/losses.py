import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import random


def get_loss_fn(loss_fn_name, weight=None, smoothing=0.05):
    loss_fn_dict = {
        'crossentropy': nn.CrossEntropyLoss(weight=weight),
        'focalloss': FocalLoss(),
        'focalcosineloss': FocalCosineLoss(weight=weight, smoothing=smoothing),
        'labelsmoothing': MyLabelSmoothingLoss(smoothing=smoothing),
        'bitemperedloss': BiTemperedLoss(t1=0.5, t2=1.0, smoothing=smoothing),
        'focalcosinebitemper': FocalCosineBiTemperLoss(weight=weight, smoothing=smoothing)
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
    def __init__(self, weight=None, alpha=1, gamma=2, xent=.1, classes=5, smoothing=0.0, p=0.5):
        super(FocalCosineLoss, self).__init__()
        self.weight = weight
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

        cosine_loss = F.cosine_embedding_loss(F.normalize(input), t, self.y, reduction=reduction)
        cent_loss = F.cross_entropy(input, target, weight=self.weight, reduce=False)

        # cosine_loss = F.cosine_embedding_loss(input, t, self.y, reduction=reduction)
        # cent_loss = F.cross_entropy(F.normalize(input), target, weight=self.weight, reduce=False)
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
# Code taken from https://github.com/fhopfmueller/bi-tempered-loss-pytorch/blob/master/bi_tempered_loss_pytorch.py

def log_t(u, t):
    """Compute log_t for `u'."""
    if t==1.0:
        return u.log()
    else:
        return (u.pow(1.0 - t) - 1.0) / (1.0 - t)

def exp_t(u, t):
    """Compute exp_t for `u'."""
    if t==1:
        return u.exp()
    else:
        return (1.0 + (1.0-t)*u).relu().pow(1.0 / (1.0 - t))

def compute_normalization_fixed_point(activations, t, num_iters):

    """Returns the normalization value for each example (t > 1.0).
    Args:
      activations: A multi-dimensional tensor with last dimension `num_classes`.
      t: Temperature 2 (> 1.0 for tail heaviness).
      num_iters: Number of iterations to run the method.
    Return: A tensor of same shape as activation with the last dimension being 1.
    """
    mu, _ = torch.max(activations, -1, keepdim=True)
    normalized_activations_step_0 = activations - mu

    normalized_activations = normalized_activations_step_0

    for _ in range(num_iters):
        logt_partition = torch.sum(
            exp_t(normalized_activations, t), -1, keepdim=True)
        normalized_activations = normalized_activations_step_0 * \
                                 logt_partition.pow(1.0-t)

    logt_partition = torch.sum(
        exp_t(normalized_activations, t), -1, keepdim=True)
    normalization_constants = - log_t(1.0 / logt_partition, t) + mu

    return normalization_constants

def compute_normalization_binary_search(activations, t, num_iters):

    """Returns the normalization value for each example (t < 1.0).
    Args:
      activations: A multi-dimensional tensor with last dimension `num_classes`.
      t: Temperature 2 (< 1.0 for finite support).
      num_iters: Number of iterations to run the method.
    Return: A tensor of same rank as activation with the last dimension being 1.
    """

    mu, _ = torch.max(activations, -1, keepdim=True)
    normalized_activations = activations - mu

    effective_dim = \
        torch.sum(
            (normalized_activations > -1.0 / (1.0-t)).to(torch.int32),
            dim=-1, keepdim=True).to(activations.dtype)

    shape_partition = activations.shape[:-1] + (1,)
    lower = torch.zeros(shape_partition, dtype=activations.dtype, device=activations.device)
    upper = -log_t(1.0/effective_dim, t) * torch.ones_like(lower)

    for _ in range(num_iters):
        logt_partition = (upper + lower)/2.0
        sum_probs = torch.sum(
            exp_t(normalized_activations - logt_partition, t),
            dim=-1, keepdim=True)
        update = (sum_probs < 1.0).to(activations.dtype)
        lower = torch.reshape(
            lower * update + (1.0-update) * logt_partition,
            shape_partition)
        upper = torch.reshape(
            upper * (1.0 - update) + update * logt_partition,
            shape_partition)

    logt_partition = (upper + lower)/2.0
    return logt_partition + mu

class ComputeNormalization(torch.autograd.Function):
    """
    Class implementing custom backward pass for compute_normalization. See compute_normalization.
    """
    @staticmethod
    def forward(ctx, activations, t, num_iters):
        if t < 1.0:
            normalization_constants = compute_normalization_binary_search(activations, t, num_iters)
        else:
            normalization_constants = compute_normalization_fixed_point(activations, t, num_iters)

        ctx.save_for_backward(activations, normalization_constants)
        ctx.t=t
        return normalization_constants

    @staticmethod
    def backward(ctx, grad_output):
        activations, normalization_constants = ctx.saved_tensors
        t = ctx.t
        normalized_activations = activations - normalization_constants
        probabilities = exp_t(normalized_activations, t)
        escorts = probabilities.pow(t)
        escorts = escorts / escorts.sum(dim=-1, keepdim=True)
        grad_input = escorts * grad_output

        return grad_input, None, None

def compute_normalization(activations, t, num_iters=5):
    """Returns the normalization value for each example.
    Backward pass is implemented.
    Args:
      activations: A multi-dimensional tensor with last dimension `num_classes`.
      t: Temperature 2 (> 1.0 for tail heaviness, < 1.0 for finite support).
      num_iters: Number of iterations to run the method.
    Return: A tensor of same rank as activation with the last dimension being 1.
    """
    return ComputeNormalization.apply(activations, t, num_iters)

def tempered_sigmoid(activations, t, num_iters = 5):
    """Tempered sigmoid function.
    Args:
      activations: Activations for the positive class for binary classification.
      t: Temperature tensor > 0.0.
      num_iters: Number of iterations to run the method.
    Returns:
      A probabilities tensor.
    """
    internal_activations = torch.stack([activations,
                                        torch.zeros_like(activations)],
                                       dim=-1)
    internal_probabilities = tempered_softmax(internal_activations, t, num_iters)
    return internal_probabilities[..., 0]


def tempered_softmax(activations, t, num_iters=5):
    """Tempered softmax function.
    Args:
      activations: A multi-dimensional tensor with last dimension `num_classes`.
      t: Temperature > 1.0.
      num_iters: Number of iterations to run the method.
    Returns:
      A probabilities tensor.
    """
    if t == 1.0:
        return activations.softmax(dim=-1)

    normalization_constants = compute_normalization(activations, t, num_iters)
    return exp_t(activations - normalization_constants, t)

def bi_tempered_binary_logistic_loss(activations,
                                     labels,
                                     t1,
                                     t2,
                                     label_smoothing = 0.0,
                                     num_iters=5,
                                     reduction='mean'):

    """Bi-Tempered binary logistic loss.
    Args:
      activations: A tensor containing activations for class 1.
      labels: A tensor with shape as activations, containing probabilities for class 1
      t1: Temperature 1 (< 1.0 for boundedness).
      t2: Temperature 2 (> 1.0 for tail heaviness, < 1.0 for finite support).
      label_smoothing: Label smoothing
      num_iters: Number of iterations to run the method.
    Returns:
      A loss tensor.
    """
    internal_activations = torch.stack([activations,
                                        torch.zeros_like(activations)],
                                       dim=-1)
    internal_labels = torch.stack([labels.to(activations.dtype),
                                   1.0 - labels.to(activations.dtype)],
                                  dim=-1)
    return bi_tempered_logistic_loss(internal_activations,
                                     internal_labels,
                                     t1,
                                     t2,
                                     label_smoothing = label_smoothing,
                                     num_iters = num_iters,
                                     reduction = reduction)

def bi_tempered_logistic_loss(activations,
                              labels,
                              t1,
                              t2,
                              label_smoothing=0.0,
                              num_iters=5,
                              reduction = 'mean',
                              p=0.5):

    """Bi-Tempered Logistic Loss.
    Args:
      activations: A multi-dimensional tensor with last dimension `num_classes`.
      labels: A tensor with shape and dtype as activations (onehot),
        or a long tensor of one dimension less than activations (pytorch standard)
      t1: Temperature 1 (< 1.0 for boundedness).
      t2: Temperature 2 (> 1.0 for tail heaviness, < 1.0 for finite support).
      label_smoothing: Label smoothing parameter between [0, 1). Default 0.0.
      num_iters: Number of iterations to run the method. Default 5.
      reduction: ``'none'`` | ``'mean'`` | ``'sum'``. Default ``'mean'``.
        ``'none'``: No reduction is applied, return shape is shape of
        activations without the last dimension.
        ``'mean'``: Loss is averaged over minibatch. Return shape (1,)
        ``'sum'``: Loss is summed over minibatch. Return shape (1,)
    Returns:
      A loss tensor.
    """

    if len(labels.shape)<len(activations.shape): #not one-hot
        labels_onehot = torch.zeros_like(activations)
        labels_onehot.scatter_(1, labels[..., None], 1)
    else:
        labels_onehot = labels

    if label_smoothing > 0 and random.random() > p:
        num_classes = labels_onehot.shape[-1]
        labels_onehot = ( 1 - label_smoothing * num_classes / (num_classes - 1) ) \
                        * labels_onehot + \
                        label_smoothing / (num_classes - 1)

    probabilities = tempered_softmax(activations, t2, num_iters)

    loss_values = labels_onehot * log_t(labels_onehot + 1e-10, t1) \
                  - labels_onehot * log_t(probabilities, t1) \
                  - labels_onehot.pow(2.0 - t1) / (2.0 - t1) \
                  + probabilities.pow(2.0 - t1) / (2.0 - t1)
    loss_values = loss_values.sum(dim = -1) #sum over classes

    if reduction == 'none':
        return loss_values
    if reduction == 'sum':
        return loss_values.sum()
    if reduction == 'mean':
        return loss_values.mean()


class BiTemperedLoss(nn.Module):
    def __init__(self, t1, t2, smoothing, p=0.5):
        super(BiTemperedLoss, self).__init__()
        self.t1 = t1
        self.t2 = t2
        self.smoothing = smoothing
        self.p = p

    def forward(self, pred, label):
        # label = F.one_hot(label, num_classes=5).float()

        return bi_tempered_logistic_loss(pred, label, self.t1, self.t2, self.smoothing, num_iters=5, p=self.p)


class FocalCosineBiTemperLoss(nn.Module):
    def __init__(self, weight=None, alpha=1, gamma=2, xent=.1, classes=5, smoothing=0.0, p=0.5, t1=0.5, t2=1.0):
        super(FocalCosineBiTemperLoss, self).__init__()
        self.focal_loss = FocalCosineLoss(weight, alpha, gamma, xent, classes, smoothing, p)
        self.bitemperloss = BiTemperedLoss(t1, t2, smoothing, p)

    def forward(self, pred, label):
        f = self.focal_loss(pred, label)
        b = self.bitemperloss(pred, label)

        return 0.7 * f + 0.3 * b

