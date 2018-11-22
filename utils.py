import numpy as np
from PIL import Image
import os

# img=Image.open('1.jpg')
# array=np.array(img)  #uint8
# print(array.dtype,array.shape)
def mkdir(file):
    if not os.path.exists(file):
        os.makedirs(file)

def imwrite(image, path):
    """ save an [-1.0, 1.0] image """

    if image.ndim == 3 and image.shape[2] == 1:  # for gray image
        image = np.array(image, copy=True)
        image.shape = image.shape[0:2]

    imgarray=((image+1.0)*127.5).astype(np.uint8)
    img=Image.fromarray(imgarray)
    img.save(path)



def immerge(images, row, col):
    """
    merge images into an image with (row * h) * (col * w)

    `images` is in shape of N * H * W(* C=1 or 3)
    """

    h, w = images.shape[1], images.shape[2]
    if images.ndim == 4:
        img = np.zeros((h * row, w * col, images.shape[3]))
    elif images.ndim == 3:
        img = np.zeros((h * row, w * col))
    for idx, image in enumerate(images):
        i = idx % col
        j = idx // col
        img[j * h:j * h + h, i * w:i * w + w, ...] = image

    return img