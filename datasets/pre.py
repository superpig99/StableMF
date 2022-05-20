import os
import shutil

filepath = "./split_compositional_dominant_sketch_target_photo/"
types = ["art_painting", "cartoon", "photo", "sketch"]
classes = ["", "dog", "elephant", "giraffe", "guitar", "horse", "house", "person"]

# train
for type in types:
    with open(filepath + type + "_train.txt", "r") as fr:
        source_classes = fr.readlines()
        for source_class in source_classes:
            source_class = source_class.strip().split(' ')
            s = source_class[0]
            c = classes[int(source_class[1])]
            filename = s.replace("/", "-")
            source = "PACS/" + s
            dest = "train/" + c + "/" + filename
            shutil.copy(source, dest)
        fr.close()

# test
for type in types:
    with open(filepath + type + "_test.txt", "r") as fr:
        source_classes = fr.readlines()
        for source_class in source_classes:
            source_class = source_class.strip().split(' ')
            s = source_class[0]
            c = classes[int(source_class[1])]
            filename = s.replace("/", "-")
            source = "PACS/" + s
            dest = "test/" + c + "/" + filename
            shutil.copy(source, dest)
        fr.close()

# val
for type in types:
    with open(filepath + type + "_val.txt", "r") as fr:
        source_classes = fr.readlines()
        for source_class in source_classes:
            source_class = source_class.strip().split(' ')
            s = source_class[0]
            c = classes[int(source_class[1])]
            filename = s.replace("/", "-")
            source = "PACS/" + s
            dest = "val/" + c + "/" + filename
            shutil.copy(source, dest)
        fr.close()