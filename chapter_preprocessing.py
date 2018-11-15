import PyPDF2
from nltk import word_tokenize, sent_tokenize, pos_tag

def pdf_reader(pdf_file_path):  # returns sentences from pdf
    # creating a pdf file object
    pdfFileObj = open(pdf_file_path, 'rb')

    # creating a pdf reader object
    pdfReader = PyPDF2.PdfFileReader(pdfFileObj)

    # number of pages in pdf file
    pages = pdfReader.numPages

    text =""
    # creating a page object
    # for i in pages:
    #     pageObj = pdfReader.getPage(i)
    pageObj = pdfReader.getPage(0)
    # extracting text from page
    text = pageObj.extractText()
    print(word_tokenize(text))
    # closing the pdf file object
    pdfFileObj.close()


with open('data/IE_chapter17.txt', 'r') as f:
    text = f.read()
work_sentence = sent_tokenize(text)[168]

print(pos_tag(word_tokenize(work_sentence)))