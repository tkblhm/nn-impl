from utils.data_generator import FontGenerator

import numpy as np



if __name__ == '__main__':
    font_generator = FontGenerator(range(30, 60, 2))
    # font_generator.generate_and_append_dataset(["utils/fonts/cour.ttf", "utils/fonts/courbd.ttf", "utils/fonts/courbi.ttf", "utils/fonts/couri.ttf"], ["utils/fonts/comic.ttf", "utils/fonts/comicbd.ttf", "utils/fonts/comici.ttf", "utils/fonts/comicz.ttf"], 10, False, "utils/pics")
    # font_generator.save_arrays("resources/arrays.npz")
    font_generator.load_arrays("resources/arrays.npz")
    print(np.info(font_generator.X))
    print(np.info(font_generator.y))
    print( )