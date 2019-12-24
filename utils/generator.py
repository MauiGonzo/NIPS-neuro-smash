import os

import PIL.Image as Image

from utils.transformer import Transformer
import Neurosmash


def generate_images(num_images=1000, start_frame=100, image_period=25):
    """Generate images of the environment's state through multiple episodes.

    Args:
        num_images   = [int] number of images to generate
        start_frame  = [int] number of frames, from episode onset, before
                             generating new images
        image_period = [int] number of frames between two images
    """
    # make directories for images
    if not os.path.isdir(f'{data_dir}images'):
        os.makedirs(f'{data_dir}images')
    if not os.path.isdir(f'{data_dir}label_images'):
        os.makedirs(f'{data_dir}label_images')

    while num_images > 0:
        end, reward, state_img = environment.reset()

        frame = 0
        while not end and num_images > 0:
            action = agent.step(end, reward, state_img)
            end, reward, state_img = environment.step(action)
            frame += 1

            if frame >= start_frame and frame % image_period == 0:
                # make normal and perspective transformed images
                img_untransformed = environment.state2image(state_img)
                img_transformed = transformer.perspective(img_untransformed)

                # make label image that combines the two images
                img = Image.new('RGB', (2*transformer.size, transformer.size))
                img.paste(img_transformed, (0, 0))
                img.paste(img_untransformed, (transformer.size, 0))

                # save images to image directories
                img_transformed.save(f'{data_dir}images/{num_images}.png')
                img.save(f'{data_dir}label_images/{num_images}.png')

                num_images -= 1


if __name__ == '__main__':
    data_dir = '../data/'
    size = 64
    timescale = 10

    environment = Neurosmash.Environment(size=size, timescale=timescale)
    agent = Neurosmash.Agent()

    transformer = Transformer(
        size, bg_file_name=f'{data_dir}background_transposed_64.png'
    )
    generate_images()
