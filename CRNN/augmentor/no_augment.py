
import imgaug as ia
import imgaug.augmenters as iaa


class MyAugmentor(object):
    def __init__(self):
        sometimes = lambda aug: iaa.Sometimes(0.5, aug)
        self.seq = iaa.Sequential([
            iaa.Multiply((0.6, 3.5), per_channel=0.5),
            sometimes(iaa.PerspectiveTransform(scale=(0.01, 0.08))),
            sometimes(iaa.AdditiveGaussianNoise((0.02, 0.2))),
            sometimes(iaa.AdditivePoissonNoise((0.02,0.1)))
        ])
