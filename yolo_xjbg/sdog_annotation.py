import xml.etree.ElementTree
import sys
import os

import tqdm

classes = ["scottish_deerhound", "silky_terrier"]
class2id = dict(zip(classes, range(len(classes))))

def parse_annotation(path):
    xml_root = xml.etree.ElementTree.parse(path).getroot()
    object = xml_root.findall('object')[0]
    name = object.findall('name')[0].text
    bound_box = object.findall('bndbox')[0]
    fn = os.path.join(xml_root.findall('folder')[0].text.lower()+"-"+name, xml_root.findall('filename')[0].text.lower() +".jpg")
    return (
        path.replace("Annotation","Images")+'.jpg'
        , bound_box.findall('xmin')[0].text
        , bound_box.findall('ymin')[0].text
        , bound_box.findall('xmax')[0].text
        , bound_box.findall('ymax')[0].text
        , class2id.get(name.lower())
    )

def main(argv):
    input, output_fn = argv

    images_root_dir = os.path.join(input, 'Images')
    annotations_root_dir = os.path.join(input, 'Annotation')
    with open(output_fn, "w") as output_file:
        for breed_dir in tqdm.tqdm([d for d in os.listdir(annotations_root_dir)]):
            # print(breed_dir)
            for annotation_file in [f for f in os.listdir(os.path.join(annotations_root_dir, breed_dir))]:
                annotation = parse_annotation(os.path.join(annotations_root_dir, breed_dir, annotation_file))
                if annotation[-1] is not None:
                    output_file.write("{} {},{},{},{},{}\n".format(*annotation))

if __name__ == '__main__':
    main(sys.argv[1:])
