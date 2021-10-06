import torch

def compute_iou(targets, predictions, eps = 1e-6):
    targets = targets.int()
    predictions = predictions.int()

    intersection = (targets & predictions).float().sum((1,2))
    union = (targets | predictions).float().sum((1,2))

    iou = (intersection + eps) / (union + eps)

    return iou.mean()

def compute_dice(targets, predictions, eps = 1e-6):
    targets = targets.int()
    predictions = predictions.int()

    intersection = (targets & predictions).float().sum((1,2))
    union = (targets | predictions).float().sum((1,2))

    dice = 2 * (intersection + eps) / (union + intersection + eps)

    return dice.mean()


def compute_accuracy(targets, predictions):
    targets = targets.int()
    predictions = predictions.int()

    batch_size, H, W = targets.shape

    targets = targets.view(batch_size, -1)
    predictions = predictions.view(batch_size, -1)

    acc = (targets == predictions).float().sum(1) / (H * W)

    return acc.mean()


if __name__ == '__main__':
    x = torch.ones(5,256,256)
    y = torch.zeros(5, 256, 256)

    print("IOU: ", compute_iou(x, y))
    print('Dice: ', compute_dice(x, y))
    print('Accuracy: ', compute_accuracy(x,y))

