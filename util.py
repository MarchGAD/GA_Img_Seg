import torch

def cal_fitness(img, thresh):
    mask1 = img < thresh
    mask2 = img >= thresh
    # 计算各类别数量 -> hist(x)
    num1 = torch.sum(mask1, dtype=torch.float32) + 1e-3
    num2 = torch.sum(mask2, dtype=torch.float32) + 1e-3
    # 计算各类别的均值
    mean1 = torch.sum(img[mask1]) / num1
    mean2 = torch.sum(img[mask2]) / num2
    p1 = num1 / (num1 + num2)
    p2 = 1 - p1
    # 加权计算全局均值
    mean_global = p1 * mean1 + p2 * mean2
    # 计算类间方差
    Sbetween = p1 * (mean1 - mean_global) ** 2 + p2 * (mean2 - mean_global) ** 2
    # 计算类内方差
    S1 = torch.sum((img[mask1] - mean1) ** 2)
    S2 = torch.sum((img[mask2] - mean2) ** 2)
    Si = S1 + S2
    Sg = torch.sum((img - mean_global) ** 2)
    Swithin = Si / Sg
    return Sbetween / Swithin


def rolletwheel(threshes, fits):
    summ = float(sum(fits))
    # 将所有适应度进行归一化
    norm_fits = [i / summ for i in fits]
    accu_probs = [0]
    # 计算累积概率
    for prob in norm_fits:
        accu_probs.append(accu_probs[-1] + prob)
    accu_probs.pop(0)
    # 生成随机值，由于该算法每次仅生成两个阈值，为此仅生成两个随机值用于计算
    chosen_probs = torch.rand(2)
    chosen_thresh = []
    for prob in chosen_probs:
        for cnt, accu_prob in enumerate(accu_probs):
            if accu_prob > prob:
                chosen_thresh.append(threshes[cnt])
                break
    return chosen_thresh


def bin2dec(bin):
    binstring = ''.join([i for i in bin])
    return int(binstring, 2)

def dec2bin(dec):
    return [i for i in str(bin(dec))[2:]]

if __name__ == '__main__':
    rolletwheel([8, 9], [7, 100])
