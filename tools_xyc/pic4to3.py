from PIL import Image
import os

root='/home/tj816/lyc/lyc/005139A/d1'

for pic in os.listdir(root):
    im = Image.open(os.path.join(root,pic))
    a=im.split()
    im=Image.merge('RGB',(a[0:3]))
    # im.show()
    im.save(os.path.join(root,pic))



