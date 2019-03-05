import spacy
from spacy import displacy

nlp = spacy.load('en_core_web_sm')

doc1 = nlp(u'For example , we know that Earth is about 4.6 billion years old and that the dinosaurs became extinct about 65 million years ago .')
doc2 = nlp(u'So , by adding the protons and neutrons in an atom \u2019s nucleus , we derive the atom \u2019s mass number .')
doc3 = nlp(u'As discussed in Chapter 1 , dates expressed in millions and billions of years truly stretch our imagination because our personal calendars involve time measured in hours , weeks , and years ')
doc4 = nlp(u'To summarize with an example , uranium \u2019s nucleus always has 92 protons , so its atomic number is always 92 .')
doc5 = nlp(u'Chemical Differentiation and Earth’s Layers The early period of heating resulted in another process of chemical differentiation, whereby melting formed')

with open('visual.html','w') as f:
    f.write(displacy.render(doc5, style='dep'))
    # f.write(displacy.serve(doc5, style='ent'))


