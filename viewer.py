import numpy as np
from PIL import Image

def viewer(a, b, show = True):
    c = np.concatenate((a, b), axis = 1)
    im = Image.fromarray(c.astype(np.uint8))
    if show:
        im.show()
