import torch

def calculate_diff(img, thresh):
    mask1 = img > thresh
    mask2 = img <= thresh
    inner_std1 = torch.std(img[mask1])
    inner_std2 = torch.std(img[mask2])
    between_dif = (torch.mean(img[mask1]) - torch.mean(img[mask2])) ** 2
    return (inner_std1 + inner_std2) / between_dif

def cal_fitness(img, threshes, L=256):
    L = torch.log2(L)
    n = len(threshes)
