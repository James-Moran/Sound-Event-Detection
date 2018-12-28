import numpy
import vggish_input
from os import listdir
from os.path import isdir, join
from pickle import dump, load

# This file is to read in the data into the correct formats
# Assuming data is made of samples within label folders, all at the given path

def parse_features_labels(path, savefile=""):
    # Get classes from the names of the subdirectories
    classes = [i for i in listdir(path) if isdir(join(path, i))]
    num_classes = len(classes)
    features, labels = [], []

    print(classes)
    for i, dir in enumerate(classes):
        # Create one hot encoding for label
        label = [0] * num_classes
        label[i] = 1
        # label = i

        # Iterate through all files in the label directory
        for f in listdir(join(path, dir)):
            try:
                # Extract features from .wav files
                fts = vggish_input.wavfile_to_examples(join(path, dir, f))
                if(len(fts) == 0):
                    continue
                fts = numpy.expand_dims(fts, axis=3)
                features.append(fts)
                labels.append(label)
                print(fts.shape)
                print(f)

            except ValueError as e:
                print(e)
                print("Not .wav format: " + f)

    features = numpy.array(features)
    labels = numpy.array(labels)
    # Save the extracted data if location given
    if savefile:
        with open(savefile, "wb") as f:
            dump((classes, features, labels), f)

    return (classes, features, labels)

# Load class names, features and labels from previous extraction
def load_features_labels(path):
    with open(path, "rb") as f:
        classes, features, labels = load(f)
    return (classes, features, labels)


if __name__ == "__main__":
    # To test
    get_features_labels("/cs/home/jm361/SH/test/")
