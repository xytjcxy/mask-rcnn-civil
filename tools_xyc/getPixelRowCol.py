from PIL import Image
import os
import numpy as np


def check(pic):
    pink, green, red, bk = 0, 0, 0, 0
    for row in pic:
        for pixel in row:
            if (pixel == [0, 0, 0]).all() or (pixel == [255, 255, 255]).all():
                bk += 1
                continue
            a, b, c = pixel[0], pixel[1], pixel[2]
            a1, a2 = a / b, b / a
            if a1 >= 1.7 and a1 <= 2.5:
                pink += 1
            elif a1 >= 5:
                red += 1
            elif a2 >= 5:
                green += 1
            else:
                bk += 1
    return (pink, green, red, bk)


if __name__ == '__main__':
    Root = "/home/tj816/Mask_RCNN-master"
    img_floder = os.path.join(Root, "gtVStest")
    pink, green, red = 0, 0, 0
    res = []
    for pic in os.listdir(img_floder):
        pic = os.path.join(img_floder, pic)
        img = Image.open(pic)
        im_array = np.array(img)
        res.append(check(im_array))
