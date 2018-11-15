import codecs
import os, subprocess
from config import Config

from nltk import word_tokenize, sent_tokenize, pos_tag


def pdf2text(pdf_file_path):  # converts pdf file to txt file
    env = dict(os.environ)
    env['LC_ALL'] = 'en_US.UTF-8'
    out_fname = os.path.splitext(os.path.basename(pdf_file_path))[0]
    cmd = ['pdftotext', pdf_file_path, os.path.join(os.path.dirname(os.path.abspath(pdf_file_path)), out_fname+'.txt')]
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, stdin=subprocess.PIPE)
    out, err = p.communicate()
    with codecs.open(os.path.join(out_fname+'.txt'), 'r', encoding='utf-8') as f_in:
        content = f_in.read()
    content.replace('\r', '').replace('\x0C', '')
    return content

if __name__ == '__main__':
    pdf2text('data/IE_chapter17.pdf')

# work_sentence = sent_tokenize(text)[168]

# print(pos_tag(word_tokenize(work_sentence)))