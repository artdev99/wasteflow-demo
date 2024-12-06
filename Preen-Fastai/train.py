from fastai.data.transforms import get_image_files, Normalize
import numpy as np
import matplotlib
import os.path

import cv2
import sys
np.set_printoptions(threshold=sys.maxsize)

from fastai.vision.augment import aug_transforms

from fastai.vision.data import SegmentationDataLoaders, Path, imagenet_stats
from fastai.vision.all import unet_learner, accuracy, RandomErasing, RandomCrop, RandomResizedCrop, Hue, Saturation
from torchvision.models import resnet18, resnet34, resnet152, wide_resnet101_2

MAGENTA = [255,0,255]

# python setup.py bdist_wheel
path = Path("/home/eco/aix/fastai/Detailed/dev/")
codes = np.loadtxt(path / 'codes.txt', dtype=str)
fnames = get_image_files(path / "images_windows")
name2id = {v: k for k, v in enumerate(codes)}
void_codes = [[name2id['a'], name2id['b']]]


def label_func(fn): return path / "labels_windows" / f"{fn.stem}.png"


epochs = 50
bs = 4
val = 0.03
arch = resnet34

dls = SegmentationDataLoaders.from_label_func(
    path, bs=bs, fnames=fnames, label_func=label_func, codes=codes,
    valid_pct=val,
    batch_tfms=[*aug_transforms(size=300, do_flip=True),
                Normalize.from_stats(*imagenet_stats),
                RandomResizedCrop(size=300),
                RandomErasing(max_count=1)
                ]
)


learn = unet_learner(dls, arch)
learn.load("/home/eco/aix/fastai/Detailed/dev/models/detailed-0.004-50-4-3.0-resnet34-fit-nf")

fnames = get_image_files(path / "Set_Testing_resized")
save_path = "/home/eco/aix/fastai/Detailed/dev/output_windows"
save_path = save_path + "/"

#use opencv to color the predicted images
#display original and labeled by NN side by side

for i in range(len(fnames)):
    img = cv2.imread(str(fnames[i]))
    predicted = learn.predict(img)[1]
    predicted = np.array(predicted)
    img2 = img.copy()
    rows,cols,dim = img2.shape
    filename = os.path.basename(str(fnames[i]))
    filename = filename[:-4]
    print(filename)
    for i in range(rows):
        for j in range(cols):
                if predicted[i,j] == name2id['windows']:
                    img2[i,j] = MAGENTA

    cv2.imwrite(save_path+filename+".jpg",img)
    cv2.imwrite(save_path+filename+".png",img2)

"""

#picture = "/home/eco/PycharmProjects/open3d/car.jpg"
#picture = "/home/eco/aix/fastai/images/00024.jpg"
picture = "/home/eco/aix/fastai/Detailed/dev/Set_Testing_resized/Lambo1.jpg"
#picture = "/home/eco/PycharmProjects/open3d/car_hq.jpg"
#picture = "/home/eco/aix/tmp/jungle.jpg"
#picture = "/home/eco/aix/fastai/front_door/00014.jpg"
image = cv2.imread(picture)
output = learn.predict(image)[1]
print(output.shape)


for r in range(0, 300):
    for c in range(0, 300):
        print(output[r][c].item(), end='')
    print()


A = np.array(np.where(output == 7))
matplotlib.pyplot.imshow(image)
matplotlib.pyplot.scatter(A[1],A[0])
print(A.shape)

matplotlib.pyplot.show()

matplotlib.pyplot.imshow(image)
matplotlib.pyplot.show()

"""