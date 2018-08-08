import tensorflow as tf
import fastText
import re

def preprocess_sentence(w):
    
    # creating a space between a word and the punctuation following it
    # eg: "he is a boy." => "he is a boy ." 
    # Reference:- https://stackoverflow.com/questions/3645931/python-padding-punctuation-with-white-spaces-keeping-punctuation
    w = re.sub(r"([?.!,¿])", r" \1 ", w)
    w = re.sub(r'[" "]+', " ", w)
    
    # replacing everything with space except (a-z, A-Z, ".", "?", "!", ",")
    # w = re.sub(r"[^a-zA-Z?.!,¿]+", " ", w)
    
    w = w.rstrip().strip()
    
    # adding a start and an end token to the sentence
    # so that the model know when to start and stop predicting.
    # w = '<start> ' + w + ' <end>'
    return w

def preprocess_sparql(s):
    """
    Add start symbol and end symbol to a sparql query for decoder processing
    """

    s = re.sub(r"PREFIX\s[^\s]*\s[^\s]*", "", s)

    return "<s> "+s.rstrip().strip()+" </s>" 