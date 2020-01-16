
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
            sometimes(
                iaa.OneOf([
                    iaa.Dropout((0.02, 0.03)),
                    iaa.Salt((0.02, 0.03))
                ])
            ),
            sometimes(iaa.ChannelShuffle(1.0)),
            #sometimes(iaa.Invert(1.0, per_channel=True)),
            #sometimes(iaa.Invert(1.0)),
            #sometimes(iaa.CropAndPad(
            #    percent=(-0.1, 0.1), pad_cval=(0,255)
            #)),
            iaa.Affine(
                #scale={"x": (0.8, 1.1), "y": (0.8, 1.1)}, # scale images to 80-120% of their size, individually per axis
                rotate=(-10, 10), # rotate by -45 to +45 degrees
                mode=ia.ALL # use any of scikit-image's warping modes (see 2nd image from the top for examples)
            ),
            #sometimes(iaa.PerspectiveTransform(scale=(0.01, 0.08))),
            sometimes(iaa.AdditiveGaussianNoise((0.02, 0.1))),
            sometimes(iaa.AdditivePoissonNoise((0.02,0.1))),
            #sometimes(iaa.Pad(
            #    percent=(0, 0.15), pad_mode=["edge"]
            #))
        ])
