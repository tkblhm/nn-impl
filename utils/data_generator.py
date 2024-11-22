import numpy as np
import random
from PIL import Image, ImageDraw, ImageFont
import string
import os


class FontGenerator:
    def __init__(self):
        pass

    def generate_text_image(self, text, font_path, output_path, image_size=(64, 64), font_size=40):
        image = Image.new("L", image_size, "white")
        draw = ImageDraw.Draw(image)

        font = ImageFont.truetype(font_path, font_size)

        w, h = draw.textbbox(text, font=font)

    def generate_flattened_chars_dataset(self, output_file, font_path, font_size, num_samples=20, image_size=(64, 64), to_image=False, image_path=""):
        """
        Generates a dataset of single letters rendered in multiple fonts and saves as an .npz file.

        Parameters:
            output_file (str): Path to save the .npz file.
            fonts_dir (str): Path to the directory containing font files (.ttf or .otf).
            image_size (tuple): Size of each image (width, height).
            num_samples (int): Number of samples per letter per font.
            grayscale (bool): Convert images to grayscale if True, otherwise keep RGB.

        Returns:
            None
        """

        letters = string.printable[:94]


        image_data = []

        font = ImageFont.truetype(font_path, size=font_size)

        for letter in letters:
            for i in range(num_samples):
                # Create a blank image
                image = Image.new("L", image_size, "white")
                draw = ImageDraw.Draw(image)

                # Load font and calculate size to center the letter
                w, h = draw.textbbox((0, 0), letter, font=font)[2:]

                # Draw the letter
                draw.text((int((image_size[0]-w)*random.random()), int((image_size[1]-h) * random.random())), letter, fill="black", font=font)

                # Convert to numpy array and normalize
                img_array = np.array(image, dtype=np.float32) / 255.0

                if to_image:
                    image.save(f"{image_path}/{letter + str(i)}.png")
                # Append to dataset
                image_data.append(img_array.flatten())



        # Convert to numpy arrays
        image_data = np.array(image_data)
        return image_data
        # Save dataset as .npz file
        # np.savez_compressed(output_file, images=image_data, labels=labels, label_map=label_map)
        # print(f"Dataset saved to {output_file}")


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
    font_generator = FontGenerator()
    font_generator.generate_flattened_chars_dataset("", )
