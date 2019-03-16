import re
import os


def clean(path):
    """
    Cleaning up data after conversion pdf to txt format
    :param path: path to txt version of the book
    :return:
    """
    out_fname = os.path.splitext(os.path.basename(path))[0] + '_cleaned'
    working_dir = os.path.dirname(os.path.abspath(path)) + '/'
    with open(path, 'r') as file:
        raw0 = file.readlines()
    """deleting uppercase words"""
    raw1 = [raw0[i] for i in range(len(raw0)) if len([w for w in raw0[i].split() if w.isupper()]) == 0]
    """deleting lines with just a number in it"""
    raw2 = [raw1[i] for i in range(len(raw1)) if len(raw1[i].split()) > 1 or len(raw1[i].split()) == 0]
    """deleting single lines with \n prev and after them"""
    raw3 = [raw2[i] for i in range(1, len(raw2) - 1) if len(raw2[i - 1].split()) != 0 or len(raw2[i + 1].split()) != 0]
    """deleting txt blocks ending with no punct and \n after them"""
    # with open('res3.txt', 'w') as f:
    #     for l in raw3:
    #         f.write(''.join(l))
    start = False
    block = []
    raw4 = []
    i = -1
    for l in raw3:
        i += 1
        if len(l.split()) != 0:
            l = re.sub(r"(\([ ,A-z]*Figure[ ,.,0-9]*\))", '', l)
            l = re.sub(r'(^Figure[ , ., 0-9]*)', '', l)
            l = re.sub(r'([A-z]*Figure[ , ., 0-9]*)', '', l)
            if len(l.split()) != 0:
                # if we delete Figure, we don't need to put this line
                if i < len(raw3) - 1 \
                        and len(raw3[i + 1].strip()) != 0 \
                        and raw3[i + 1][0].isupper() and not start:
                    pass
                else:
                    block.append(l.strip() + ' ')
                    start = True
        if len(l.split()) == 0 and start and len(block) > 0:  # we met end of block
            if str(block[-1].strip()[-1]) in ['.', '?', '!']:  # if block ends with . ? valid block
                raw4.append(block)
                raw4.append('\n\n')
                start = False
                block = []
            else:
                if i < len(raw3)-1 \
                        and (len(raw3[i+1].strip()) == 0 or raw3[i+1][0].isupper()):
                    start = False
                    block = []
    # ( \(Figure[ , . ,0-9]*\)) deleting (Figure ....)

    with open(working_dir + out_fname + '.txt', 'w') as f:
        for l in raw4:
            f.write(''.join(l))


if __name__ == '__main__':
    clean('../data/Books/Microbiology_full.txt')
