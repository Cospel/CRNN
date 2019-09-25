
import imgaug as ia
import imgaug.augmenters as iaa


class MyAugmentor(object):
    def __init__(self):
        sometimes = lambda aug: iaa.Sometimes(0.5, aug)
        self.seq = iaa.Sequential([
            iaa.Multiply((0.8, 1.2), per_channel=0.5),
            #sometimes(iaa.PerspectiveTransform(scale=(0.01, 0.05))),
            sometimes(
                iaa.OneOf([
                    iaa.CoarseDropout((0.01, 0.03), size_percent=(0.1, 0.3)),
                    iaa.CoarseDropout((0.01, 0.03), size_percent=(0.1, 0.3), per_channel=1.0),
                    iaa.Dropout((0.03,0.05)),
                    iaa.Salt((0.03,0.05))
                ])
            ),
            #iaa.Multiply((0.8, 1.2), per_channel=0.5),
            sometimes(iaa.FrequencyNoiseAlpha(
                    exponent=(-4, 0),
                    first=iaa.Multiply((0.8, 1.2), per_channel=0.5),
                    second=iaa.ContrastNormalization((0.8, 1.5))
                )
            ),
            sometimes(
                iaa.OneOf([
                    iaa.MotionBlur(k=(3,4),angle=(0, 360)),
                    iaa.GaussianBlur((0, 1.2)),
                    iaa.AverageBlur(k=(2, 3)),
                    iaa.MedianBlur(k=(3, 5))
                ])
            ),
            #sometimes(
            #    iaa.CropAndPad(
            #        percent=(-0.05, 0.1),
            #        pad_mode='constant',
            #        pad_cval=(0, 255)
            #    ),
            #),
            #sometimes(iaa.ElasticTransformation(alpha=(1.0, 2.0), sigma=(2.0, 3.0))), # move pixels locally around (with random strengths)
            #sometimes(iaa.PiecewiseAffine(scale=(0.01, 0.02), mode='constant')), # sometimes move parts of the image around
            sometimes(iaa.AdditiveGaussianNoise((0.02, 0.1))),
            sometimes(iaa.AdditivePoissonNoise((0.02,0.05)))
        ])