import torch

def accuracy(logits, y):
    pred = torch.log_softmax(logits, dim=1)
    correct = pred.argmax(dim=1).eq(y).sum().item()
    total = len(y)
    acc = correct / total
    return acc