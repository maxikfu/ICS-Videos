import re
import pytextrank


def clean(path):
    with open(path, 'r') as file:
        raw0 = file.readlines()
    """deleting uppercase words"""
    raw1 = [raw0[i] for i in range(len(raw0)) if len([w for w in raw0[i].split() if w.isupper()]) == 0]
    """deleting lines with just a number in it"""
    raw2 = [raw1[i] for i in range(len(raw1)) if len(raw1[i].split()) > 1 or len(raw1[i].split()) == 0]
    """deleting single lines with \n prev and after them"""
    raw3 = [raw2[i] for i in range(1, len(raw2) - 1) if len(raw2[i - 1].split()) != 0 or len(raw2[i + 1].split()) != 0]
    """deleting txt blocks ending with no punct and \n after them"""
    start = False
    block = []
    raw4 = []
    i = -1
    for l in raw3:
        i += 1
        if len(l.split()) != 0:
            start = True
            """if line ends with uppercase and next line starts with uppercase"""
            if i < len(raw3)-2 and len(raw3[i].split()) >= 2 \
                    and len(raw3[i+1].split()) != 0 \
                    and str(raw3[i].split()[-1])[0].isupper() \
                    and str(raw3[i+1].split()[0])[0].isupper():
                pass
            else:
                if any([j in ['Figure', 'SmartFigure'] for j in l.split()]):
                    pass
                else:
                    l = re.sub(r'( \(Figure[ , ., 0-9]*\))', '', l)
                    l = re.sub(r'(\(Figure[ , ., 0-9]*\))', '', l)
                    block.append(l.strip()+' ')
        if len(l.split()) == 0 and start and len(block) > 2:
            if str(block[-1].strip()[-1]) in ['.', '?']:
                raw4.append(block)
                raw4.append('\n\n')
                start = False
                block = []
            else:
                if i < len(raw3)-1 and not raw3[i+1][0].isupper():
                    pass
                else:
                    start = False
                    block = []
    # ( \(Figure[ , . ,0-9]*\)) deleting (Figure ....)
    with open('res2.txt', 'w') as f:
        for l in raw4:
            f.write(''.join(l))


if __name__ == '__main__':
    pass
