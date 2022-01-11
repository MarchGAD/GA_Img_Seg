import torch

from util import *

class GAImageSeg:

    '''
        A two-class implementation of 
        https://www.researchgate.net/publication/339595927_Multi-Thresholding_Image_Segmentation_Using_Genetic_Algorithm
    '''

    def __init__(self, cross_prob=0.85, mute_prob=0.15) -> None:
        self.threshes = [torch.randint(0, 256, (1, 1))[0][0], torch.randint(0, 256, (1, 1))[0][0]]
        
        self.cross_prob = cross_prob
        self.mute_prob = mute_prob

    @staticmethod
    def cal_fit(img, threshes):
        return [cal_fitness(img, i) for i in threshes]

    def crossover(self, thresh1, thresh2):
        bin1, bin2 = dec2bin(thresh1), dec2bin(thresh2)
        if len(bin1) < 2 or len(bin2) < 2:
            return bin1, bin2 
        cross_pos = torch.randint(1, len(bin1), (1, 1))[0][0]
        bin1 = bin1[:cross_pos] + bin2[cross_pos:]
        bin2 = bin2[:cross_pos] + bin1[cross_pos:]
        return bin2dec(bin1), bin2dec(bin2)
    
    def mutation(self, thresh):
        bin = dec2bin(thresh)
        mute_pos = torch.randint(len(bin), (1, 1))[0][0]
        bin[mute_pos] = '0' if bin[mute_pos] == '1' else '0'
        return bin2dec(bin)

    def iter(self, img, times, max_child=10):
        self.fits = self.cal_fit(img, self.threshes)
        for _ in range(times):
            new_threshes = []
            new_fitnesses = []
            cnt = 0
            while len(self.threshes) > len(new_threshes) and cnt < max_child:
                thresh1, thresh2 = rolletwheel(self.threshes, self.cal_fit(img, self.threshes))
                fits = self.cal_fit(img, [thresh1, thresh2])
                cnt += 1
                if torch.rand(1) < self.cross_prob:
                    thresh1, thresh2 = self.crossover(thresh1, thresh2)
                if torch.rand(1) < self.mute_prob:
                    thresh1 = self.mutation(thresh1)
                if torch.rand(1) < self.mute_prob:
                    thresh2 = self.mutation(thresh2)
                for thresh, fit in zip([thresh1, thresh2], self.cal_fit(img, [thresh1, thresh2])):
                    if fit > max(fits):
                        new_threshes.append(thresh)
                        new_fitnesses.append(fit)
            if cnt == max_child:
                continue
            self.threshes = new_threshes
            self.fits = new_fitnesses
        pairs = [(i, j) for i, j in zip(self.threshes, self.fits)]
        
        return sorted(pairs, key=lambda x:x[1])[-1]

if __name__ == '__main__':
    import cv2 as cv
    import numpy as np

    # img_path = './fisherman.jpg'
    img_path = './lena.jpg'

    img  = cv.imread(img_path, cv.IMREAD_GRAYSCALE)
    img = torch.from_numpy(img).float()
    ga = GAImageSeg()
    print('initial threshes are {}, {}'.format(*ga.threshes))
    thresh, fit = ga.iter(img, 5)
    print(f'thresh is {thresh}, fit is {fit}')
    mask = img > thresh
    img = 255 * mask.int().numpy().astype(np.uint8)
    b = cv.imshow('hey', img)
    cv.waitKey()

