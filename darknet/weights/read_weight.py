
# -*- coding:utf-8 -*-
import os


def get_bin_file(filename):
    '''
    Get the all content of the binary file.

    input: filename - the binary file name
    return: binary string - the content of the file.
    '''

    if not os.path.isfile(filename):
        print("ERROR: %s is not a valid file." % (filename))
        return None

    f = open(filename, "rb")
    data = f.read()
    f.close()

    return data

def main():
    weights = get_bin_file("tiny.weights")
    print weights

main()
