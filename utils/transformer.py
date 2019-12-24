import PIL.Image as Image
import numpy as np


class Transformer(object):
    """Transforms input image to control for perspective bias in distance.

    Attributes:
        size   = [int] number of width and height pixels of environment
        coeffs = [[float]] list of parameters for perspective transformation
        bg     = [ndarray] 3D array of image of background of environment
    """

    def __init__(self, size, bg_file_name='background.png'):
        """Initialize transformer.

        Args:
            size         = [int] number of width/height pixels of environment
            bg_file_name = [str] file path for background image of size x size
        """
        self.size = size

        # source pixel coordinates of corners of stage, determined by size
        source_coords = [(size / 2, -size / 40),
                         (-size / 6.25, size / 3.23),
                         (size / 2, size / 1.41),
                         (size / 0.86, size / 3.23)]

        # target pixel coordinates of corners of stage, determined by size
        target_coords = [(0, 0),
                         (0, size),
                         (size, size),
                         (size, 0)]

        # coefficients for perspective transformation, computed only once
        self.coeffs = self._find_coeffs(source_coords, target_coords)

        # save background image for filtering out agents
        self.bg = Image.open(bg_file_name).convert('RGB')
        self.bg = np.asarray(self.bg, np.uint8, 'f').transpose(2, 0, 1)

    def _find_coeffs(self, source_coords, target_coords):
        """Finds coefficients needed for perspective transformation.

        Args:
            source_coords = [[(int, int)]] source x and y pixel coordinates list
            target_coords = [[(int, int)]] target x and y pixel coordinates list

        Returns [ndarray]:
            Coefficients as a NumPy ndarray of floats.
        """
        matrix = []
        for s, t in zip(source_coords, target_coords):
            matrix.append([t[0], t[1], 1, 0, 0, 0, -s[0] * t[0], -s[0] * t[1]])
            matrix.append([0, 0, 0, t[0], t[1], 1, -s[1] * t[0], -s[1] * t[1]])
        A = np.array(matrix)
        B = np.array(source_coords).reshape(8)
        res = np.linalg.inv(A.T @ A) @ A.T @ B
        return res.reshape(8)

    def perspective(self, img):
        """"Transform the perspective of the image to get a square stage.

        Args:
            img = [Image] image of the environment's state

        Returns [Image]:
            Perspective transformed image of the environment's state,
            resulting in a square stage from a top-down perspective.
        """
        return img.transform(img.size, Image.PERSPECTIVE,
                             self.coeffs, Image.BICUBIC)

    def transpose(self, img):
        """Flip the image over the diagonal to expand the labeled data set.

        Args:
            img = [ndarray] 3D array of image of square stage

        Returns [ndarray]:
            Transpose transformed image of the square stage, resulting
            in a diagonally flipped image that can expand the data set.
        """
        img_no_bg = (img - self.bg) % 255
        img_no_bg_transposed = img_no_bg.transpose(0, 2, 1)
        return (img_no_bg_transposed + self.bg) % 255

    def flip(self, img, flip):
        """Flip the image left to right or top to bottom
           to expand the labeled data set.

        Args:
            img  = [ndarray] 3D array of image of square stage
            flip = ['left_right'|'top_bottom'] direction of flip

        Returns [ndarray]:
            Flipped image of the square stage.
        """
        img_no_bg = (img - self.bg) % 255
        if flip == 'left_right':
            img_no_bg_flipped = np.flip(img_no_bg, axis=2)
        else:  # flip == 'top_bottom':
            img_no_bg_flipped = np.flip(img_no_bg, axis=1)

        return (img_no_bg_flipped + self.bg) % 255

    def corrupt(self, img):
        """Corrupt the image by adding some Gaussian noise.

        Args:
            img  = [ndarray] 3D array of image of square stage

        Returns [ndarray]:
            Noisy or corrupted image of the square stage.
        """
        noise = np.random.normal(0, 5, (3, self.size, self.size))
        return img + noise.astype(np.uint8)

    def translate(self, img, axis, shift):
        """Translate the agents by the given direction and shift.

        Args:
            img   = [ndarray] 3D array of image of square stage
            axis  = [0|1] axis of translation; 0 is vertical and
                          1 is horizontal
            shift = [int] number of pixels to move the agents forward,
                          negative shift means the agents will be
                          moved back along the specified axis

        Returns [ndarray]:
            Image of the square stage where the agents are
            translated `shift` pixels in the given `axis`.
        """
        img_no_bg = (img - self.bg) % 255
        img_no_bg_translated = np.roll(img_no_bg, shift, axis + 1)
        return (img_no_bg_translated + self.bg) % 255
