import string
import xml.etree.ElementTree as ET
from nltk.tokenize import wordpunct_tokenize
from nltk.stem import PorterStemmer
from collections import Counter
import argparse


# XML to dict function focused on specific format
def xml_to_dict(file_path):
    """
    Takes an xml document with header 'doc' and extracts doc
    id and text. 
    :param file_path: a path to an xml document
    :return: a dictionary of {docid, text}
    """
    results = {}

    #load file
    tree = ET.parse(file_path)

    # get root
    root = tree.getroot()

    for doc in root.findall('doc'):
        doc_id = doc.findtext('docid', default='').strip()
        text = doc.findtext('msgtext', default='')

        results[doc_id] = text

    return results


# Tokenizer base class
class Tokenizer:
    """
    Informal interface for tokenizers. 
    """

    def tokenize(self, text):
        """
        Splits raw text into a list of token strings
        :param text: a string
        :return: a list of strings
        """
        raise NotImplementedError(f"{type(self).__name__} must implement tokenize()")

    def normalize(self, tokens):
        """
        Lowercases tokens and strips surrounding punctuation
        :param tokens: a list of strings
        :return: a list of strings
        """
        normalized = [t.lower().strip(string.punctuation) for t in tokens]
        return [t for t in normalized if t]

    def __call__(self, text):
        """Tokenizes, then normalizes, so a tokenizer can be used as tokenizer(text)"""
        return self.normalize(self.tokenize(text))


class WhitespaceTokenizer(Tokenizer):
    """Splits text on whitespace"""

    def tokenize(self, text):
        return text.split()


class NLTKTokenizer(Tokenizer):
    """Off-the-shelf tokenizer from nltk"""

    def tokenize(self, text):
        return wordpunct_tokenize(text)


class CharNGramTokenizer(Tokenizer):
    """Splits text into overlapping character n-grams"""

    def __init__(self, n=4):
        self.n = n

    def tokenize(self, text):
        ngrams = []
        for word in text.split():
            # if word is shorter than n, it becomes a token
            if len(word) < self.n:
                ngrams.append(word)
            # slide n-character window across every word, one character at a time
            for i in range(len(word) - self.n +1):
                ngrams.append(word[i:i + self.n])
        return ngrams

# Stem
stemmer = PorterStemmer()
def stem_words(tokens):
    """
    Takes a list of words and returns a list of stemmed words 
    :text: a list of unstemmed tokens
    :return: a list of stemmed tokens
    """
    return [stemmer.stem(token) for token in tokens]

# Index using all 3 tokenization methods
def build_index(docs, tokenizer, stem=True):
    """
    Builds an inverted index mapping each term to its postings list
    :param docs: a dictionary of {docid: text}
    :param tokenizer: a Tokenizer instance
    :param stem: whether to apply the Porter stemmer
    :return: a dictionary of {term: [(docid, term frequency), ...]}
    """
    index = {}
    for docid, text_body in docs.items():
        tokens = tokenizer(text_body)
        if stem:
            tokens = stem_words(tokens)
        term_frequency = Counter(tokens)
        for token, tf in term_frequency.items():
            if token in index:
                index[token].append((docid, tf))
            else:
                index[token] = [(docid, tf)]
    return index

def query(index, text, tokenizer, stem=True):
    """
    Returns the doc ids containing every term of the query
    :param index: an inverted index from build_index
    :param text: the query string
    :param tokenizer: the same Tokenizer used to build the index
    :param stem: whether the index was stemmed
    :return: a sorted list of doc ids
    """
    terms = tokenizer(text)
    if stem:
        terms = stem_words(terms)
    results = None
    for term in set(terms):
        docids = {docid for docid, tf in index.get(term, [])}
        results = docids if results is None else results & docids
    return sorted(results or [], key=int)

TOKENIZERS = {
    "whitespace": WhitespaceTokenizer,
    "nltk": NLTKTokenizer,
    "ngram": CharNGramTokenizer,
}

def main():
    parser = argparse.ArgumentParser(description="Inverted index for 20 Newsgroups XML")
    parser.add_argument("-f", "--file", required=True, help="Path to the XML collection")
    parser.add_argument("-t", "--tokenizer", choices=[*TOKENIZERS, "all"], default="all",
                        help="Tokenizer to use (default: all three)")
    parser.add_argument("-s", "--stem", action="store_true", help="Apply the Porter stemmer")
    args = parser.parse_args()

    docs = xml_to_dict(args.file)
    names = list(TOKENIZERS) if args.tokenizer == "all" else [args.tokenizer]
    queries = ["system", "compatibility", "system compatibility"]

    for name in names:
        tokenizer = TOKENIZERS[name]()
        index = build_index(docs, tokenizer, stem=args.stem)
        print(f"\n{name} tokenizer: {len(index)} terms, stemming {'on' if args.stem else 'off'}")
        for q in queries:
            print(f"  {q!r}: {query(index, q, tokenizer, stem=args.stem)}")

if __name__ == "__main__":
    main()