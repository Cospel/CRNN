
import imgaug as ia
import imgaug.augmenters as iaa


class MyAugmentor(object):
    def __init__(self):
        sometimes = lambda aug: iaa.Sometimes(0.5, aug)
        self.seq = iaa.Sequential([
            iaa.OneOf([
                iaa.Multiply((0.6, 1.0), per_channel=0.5),
                iaa.Multiply((1.0, 2.0), per_channel=0.5),
            ]),
            sometimes(iaa.ChannelShuffle(1.0)),
            sometimes(iaa.Invert(1.0, per_channel=True)),
            sometimes(iaa.Invert(1.0)),
            sometimes(iaa.CropAndPad(
                percent=(-0.15, 0.15), pad_cval=(0,255)
            )),
            # sometimes(iaa.Affine(
            #     rotate=(-10, 10), # rotate by -45 to +45 degrees
            #     shear=(-5, 5), # shear by -16 to +16 degrees
            #     order=[0, 1], # use nearest neighbour or bilinear interpolation (fast)
            #     cval=(0, 255), # if mode is constant, use a cval between 0 and 255
            #     mode=ia.ALL # use any of scikit-image's warping modes (see 2nd image from the top for examples)
            # )),
            sometimes(iaa.PerspectiveTransform(scale=(0.01, 0.08))),
            sometimes(iaa.AdditiveGaussianNoise((0.02, 0.2))),
            sometimes(iaa.AdditivePoissonNoise((0.02,0.1)))
        ])
