import xml.etree.ElementTree
import sys
import os

import tqdm

classes = ["hole"]
class2id = dict(zip(classes, range(len(classes))))

def parse_annotation(path):
    with open(path) as f:
        lines = f.readlines()
        num, holes = lines[0], lines[1:]
        cc = []
        for line in holes:
            left, top, right, bottom, name = line.strip().split()
            cc.append(",".join([
                left
                , top
                , right
                , bottom
                , str(class2id[name.lower()])])
            )

        return (
            path.replace(".txt", ".jpg")
            , " ".join(cc)
        )


def main(argv):
    input, output_fn = argv

    annotations_root_dir = input
    from glob import glob
    with open(output_fn, "w") as output_file:
        for breed_dir in tqdm.tqdm(os.listdir(input)):
            print(breed_dir)
            for annotation_file in glob(os.path.join(input, breed_dir, "*.txt")):
                annotation = parse_annotation(annotation_file)
                if annotation[-1] is not None:
                    output_file.write("{} {}\n".format(*annotation))


if __name__ == '__main__':
    main(sys.argv[1:])
