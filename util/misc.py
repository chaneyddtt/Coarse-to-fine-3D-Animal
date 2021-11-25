import torch
import shutil
import os


def save_checkpoint(state, is_best, checkpoint='checkpoint', filename='checkpoint.pth.tar'):

    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'model_best.pth.tar'))

def adjust_learning_rate(optimizer, epoch, lr, schedule, gamma):
    """Sets the learning rate to the initial LR decayed by schedule"""
    if epoch in schedule:
        lr *= gamma
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
    return lr


def lr_poly(base_lr, epoch, max_epoch, power):
    """ Poly_LR scheduler
    """
    return base_lr * ((1 - float(epoch) / max_epoch) ** power)


def adjust_learning_rate_main(optimizer, epoch, args):
    lr = lr_poly(args.lr, epoch, args.max_epoch, args.power)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def adjust_learning_rate_exponential(optimizer, epoch, epoch_decay, learning_rate, decay_rate):
    lr = learning_rate * (decay_rate ** (epoch / epoch_decay))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr