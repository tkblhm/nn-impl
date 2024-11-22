import os


paths = ["../nn.py", "../layers.py", "../losses.py", "../visualisations.py", "data_generator.py"]

target = "combined.txt"

with open(target, "w") as file:
    for path in paths:
        file.write("# start of " + path + "\n")
        with open(path, "r") as reader:
            file.writelines(reader.readlines())
        file.write("# end of " + path + "\n")

