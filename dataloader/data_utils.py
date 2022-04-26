import torch

from PIL import Image


extensions = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif']


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


def accimage_loader(path):
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def square_the_bbox(bbox):
    top, left, bottom, right = bbox
    width = right - left
    height = bottom - top
    if height < width:
        center = (top + bottom) * 0.5
        top = int(round(center - width * 0.5))
        bottom = top + width
    else:
        center = (left + right) * 0.5
        left = int(round(center - height * 0.5))
        right = left + height
    return top, left, bottom, right


def pil_box_loader(path, box):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        width, height = img.size

        box = square_the_bbox(box)

        topRel = box[0]
        leftRel = box[1]
        bottomRel = box[2]
        rightRel = box[3]

        # 1.1 x box
        dw = (rightRel - leftRel) * 0.3 #0.4
        dh = (bottomRel - topRel) * 0.3 #0.4

        x = max(0, leftRel - dw)
        y = max(0, topRel - dh)
        w = (rightRel - leftRel) * 1.6 #1.8
        h = (bottomRel - topRel) * 1.6 #1.8

        img = img.crop(box=(x,y,x+w,y+h))

        return img.convert('RGB')


def accimage_box_loader(path):
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def default_loader(path):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)

def box_loader(path, box):
    from torchvision import get_image_backend
    # Note that box_loader now only supports pil_loader
    return pil_box_loader(path, box)


def _setup_angle(x, name, req_sizes=(2, )):
    x = [-x, x]
    return [float(d) for d in x]



class RandomRotation(torch.nn.Module):
    def __init__(self, degrees, resample=False, expand=False, center=None, fill=None):
        super().__init__()
        self.degrees = _setup_angle(degrees, name="degrees", req_sizes=(2, ))

        if center is not None:
            _check_sequence_input(center, "center", req_sizes=(2, ))

        self.center = center

        self.resample = resample
        self.expand = expand
        self.fill = fill

    @staticmethod
    def get_params(degrees):
        """Get parameters for ``rotate`` for a random rotation.

        Returns:
            float: angle parameter to be passed to ``rotate`` for random rotation.
        """
        angle = float(torch.empty(1).uniform_(float(degrees[0]), float(degrees[1])).item())
        return angle


    def forward(self, img):
        """
        Args:
            img (PIL Image or Tensor): Image to be rotated.

        Returns:
            PIL Image or Tensor: Rotated image.
        """
        angle = self.get_params(self.degrees)
        # output = F.rotate(img, angle, self.resample, self.expand, self.center, self.fill)
        output = img.rotate(angle, self.resample, self.expand, self.center, self.fill)
        return output, angle


    def __repr__(self):
        format_string = self.__class__.__name__ + '(degrees={0}'.format(self.degrees)
        format_string += ', resample={0}'.format(self.resample)
        format_string += ', expand={0}'.format(self.expand)
        if self.center is not None:
            format_string += ', center={0}'.format(self.center)
        if self.fill is not None:
            format_string += ', fill={0}'.format(self.fill)
        format_string += ')'
        return format_string


def get_padding(image):
    w = image.size()[0]
    h = image.size()[1]
    max_wh = np.max([w, h])
    h_padding = (max_wh - w) / 2
    v_padding = (max_wh - h) / 2
    l_pad = h_padding if h_padding % 1 == 0 else h_padding+0.5
    t_pad = v_padding if v_padding % 1 == 0 else v_padding+0.5
    r_pad = h_padding if h_padding % 1 == 0 else h_padding-0.5
    b_pad = v_padding if v_padding % 1 == 0 else v_padding-0.5
    padding = (int(l_pad), int(t_pad), int(r_pad), int(b_pad))
    return padding

class NewPad(object):
    def __init__(self, fill=0, padding_mode='constant'):
        assert isinstance(fill, (numbers.Number, str, tuple))
        assert padding_mode in ['constant', 'edge', 'reflect', 'symmetric']

        self.fill = fill
        self.padding_mode = padding_mode

    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to be padded.

        Returns:
            PIL Image: Padded image.
        """
        return F.pad(img, get_padding(img), value = self.fill, mode = self.padding_mode)

    def __repr__(self):
        return self.__class__.__name__ + '(padding={0}, fill={1}, padding_mode={2})'.\
            format(self.fill, self.padding_mode)
