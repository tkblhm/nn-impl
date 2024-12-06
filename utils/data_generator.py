import numpy as np
import random
import re
from PIL import Image, ImageDraw, ImageFont
import string
import os


class FontGenerator:
    def __init__(self, font_range, image_size=(64, 64)):
        self.X = None
        self.y = None

        self.font_range = font_range
        self.image_size = image_size

    def generate_flattened_arrays_for_single_letter(self, font_path, num_samples=20, to_image=False, image_dir=""):
        """
        Generates a 2d array of flattened images of a specific font, and optionally saves as images.

        Parameters:
            font_path (str): Font path.
            num_samples (int): Number of samples per letter per font.
            to_image (bool): Whether saving as png files.
            image_dir (str): Directory of saved images.

        Returns:
            (np.ndarray): 2d array of samples of flattened array
        """

        # letters and digits and symbols
        letters = string.printable[:62]


        image_data = []
        fontname = re.search(r"(\w+)\.ttf", font_path).group(1)

        for font_size in self.font_range:
            font = ImageFont.truetype(font_path, size=font_size)


            for idx, letter in enumerate(letters):
                for i in range(num_samples):
                    # Create a blank image
                    image = Image.new("L", self.image_size, "white")
                    draw = ImageDraw.Draw(image)

                    # Load font and calculate size to center the letter
                    w, h = draw.textbbox((0, 0), letter, font=font)[2:]

                    # Draw the letter
                    draw.text((int((self.image_size[0]-w)*random.random()), int((self.image_size[1]-h) * random.random())), letter, fill="black", font=font)

                    # Convert to numpy array and normalize
                    img_array = np.array(image, dtype=np.float32) / 255.0

                    if to_image:
                        if image_dir == '' or image_dir.endswith("/"):
                            image.save(f"{image_dir}{fontname}-size-{font_size}-char-{idx}-idx-{str(i)}.png")
                        else:
                            image.save(f"{image_dir}/{fontname}-size-{font_size}-char-{idx}-idx-{str(i)}.png")
                            # image.save("pics/test.png")

                    # Append to dataset
                    image_data.append(img_array.flatten())



        # Convert to numpy arrays
        image_data = np.array(image_data)
        return image_data
        # Save dataset as .npz file
        # np.savez_compressed(output_file, images=image_data, labels=labels, label_map=label_map)
        # print(f"Dataset saved to {output_file}")

    def generate_and_append_dataset(self, font_paths0, font_paths1, num_samples=20, to_image=False, image_path=""):
        if self.X is None:
            xs = np.empty((0, self.image_size[0]*self.image_size[1]))
        else:
            xs = self.X

        if self.y is None:
            ys = np.empty((0, 1))
        else:
            ys = self.y

        for path in font_paths0:
            x = self.generate_flattened_arrays_for_single_letter(path, num_samples, to_image, image_path)
            y = np.array([0 for i in range(x.shape[0])]).reshape((-1, 1))
            xs = np.vstack((xs, x))
            ys = np.vstack((ys, y))

        for path in font_paths1:
            x = self.generate_flattened_arrays_for_single_letter(path, num_samples, to_image, image_path)
            y = np.array([1 for i in range(x.shape[0])]).reshape((-1, 1))
            xs = np.vstack((xs, x))
            ys = np.vstack((ys, y))

        self.X = xs
        self.y = ys


    def save_arrays(self, output_file):
        assert (self.X is not None and self.y is not None)
        np.savez_compressed(output_file, X=self.X, y=self.y)

    def load_arrays(self, output_file):
        data = np.load(output_file)
        self.X = data['X']
        self.y = data['y']

# n by m input and binary output
def generator(n, m, min_val, max_val, func, error=1):
    X = []
    y = []
    diff = max_val - min_val

    for i in range(n):
        x0 = [round(random.random() * diff + min_val, 2) for j in range(m)]
        X.append(x0)
        b = int(func(*x0))

        y.append(b if random.random() < error else 1-b)


    return np.array(X), np.array(y).reshape((-1,1))


if __name__ == '__main__':
    font_generator = FontGenerator(range(60, 61, 1), (64, 64))
    font_generator.generate_and_append_dataset(["fonts/cour.ttf", "fonts/courbd.ttf", "fonts/courbi.ttf", "fonts/couri.ttf"], ["fonts/comic.ttf", "fonts/comicbd.ttf", "fonts/comici.ttf", "fonts/comicz.ttf"], 1, True, "pics")
    # font_generator.generate_flattened_arrays_for_single_letter("fonts/cour.ttf", range(30, 32, 2), 1, (64, 64), True, "pics")

