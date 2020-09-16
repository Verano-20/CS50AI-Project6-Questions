import os
import sys
import string
import math
import nltk

FILE_MATCHES = 1
SENTENCE_MATCHES = 1


def main():

    # Check command-line arguments
    if len(sys.argv) != 2:
        sys.exit("Usage: python questions.py corpus")

    # Calculate IDF values across files
    files = load_files(sys.argv[1])
    file_words = {
        filename: tokenize(files[filename])
        for filename in files
    }
    file_idfs = compute_idfs(file_words)

    # Prompt user for query
    query = set(tokenize(input("Query: ")))

    # Determine top file matches according to TF-IDF
    filenames = top_files(query, file_words, file_idfs, n=FILE_MATCHES)

    # Extract sentences from top files
    sentences = dict()
    for filename in filenames:
        for passage in files[filename].split("\n"):
            for sentence in nltk.sent_tokenize(passage):
                tokens = tokenize(sentence)
                if tokens:
                    sentences[sentence] = tokens

    # Compute IDF values across sentences
    idfs = compute_idfs(sentences)

    # Determine top sentence matches
    matches = top_sentences(query, sentences, idfs, n=SENTENCE_MATCHES)
    for match in matches:
        print(match)


def load_files(directory):
    """
    Given a directory name, return a dictionary mapping the filename of each
    `.txt` file inside that directory to the file's contents as a string.
    """
    filedict = dict()
    for doc in os.listdir(directory):
        text_file = os.path.join(directory, doc)
        with open(text_file, "r", encoding="utf8") as f:
            textstring = f.read()
            filedict[doc] = textstring
        f.close()
    return filedict


def tokenize(document):
    """
    Given a document (represented as a string), return a list of all of the
    words in that document, in order.

    Process document by coverting all words to lowercase, and removing any
    punctuation or English stopwords.
    """
    words_tok = nltk.word_tokenize(document)
    words_tok = [word.lower() for word in words_tok if word not in nltk.corpus.stopwords.words("english") and word not in string.punctuation]
    return words_tok


def compute_idfs(documents):
    """
    Given a dictionary of `documents` that maps names of documents to a list
    of words, return a dictionary that maps words to their IDF values.

    Any word that appears in at least one of the documents should be in the
    resulting dictionary.
    """
    idfs = dict()
    for doc in documents:
        for word in documents[doc]:
            if word in idfs:
                continue
            word_occurences = 1
            for doc2 in documents:
                if doc2 == doc:
                    continue
                if word in documents[doc2]:
                    word_occurences += 1
            idfs[word] = math.log(len(documents) / word_occurences)
    return idfs


def top_files(query, files, idfs, n):
    """
    Given a `query` (a set of words), `files` (a dictionary mapping names of
    files to a list of their words), and `idfs` (a dictionary mapping words
    to their IDF values), return a list of the filenames of the the `n` top
    files that match the query, ranked according to tf-idf.
    """
    topfiles = dict()
    for doc in files:
        topfiles[doc] = 0
        for word in query:
            if word in files[doc]:
                tf = files[doc].count(word)
                topfiles[doc] += tf * idfs[word]
    # Sort dict by tfidf values
    topfiles = {doc: tfidf for doc, tfidf in sorted(topfiles.items(), key=lambda item: item[1], reverse=True)}
    returnfiles = []
    for i in range(n):
        returnfiles.append(list(topfiles.keys())[i])
    return returnfiles


def top_sentences(query, sentences, idfs, n):
    """
    Given a `query` (a set of words), `sentences` (a dictionary mapping
    sentences to a list of their words), and `idfs` (a dictionary mapping words
    to their IDF values), return a list of the `n` top sentences that match
    the query, ranked according to idf. If there are ties, preference should
    be given to sentences that have a higher query term density.
    """
    topsentences = dict()
    for sentence in sentences:
        topsentences[sentence] = [0, 0] # (matching word measure, query term density)
        for word in query:
            if word in sentences[sentence]:
                topsentences[sentence][0] += idfs[word]
        query_terms = 0
        for word in sentences[sentence]:
            if word in query:
                query_terms += 1
        topsentences[sentence][1] += query_terms / len(sentences[sentence])
    # Sort dict by matching word measure, then by query term density if needed
    topsentences = {sentence: values for sentence, values in sorted(topsentences.items(), key=lambda item: (item[1][0], item[1][1]), reverse=True)}
    returnsentences = []
    for i in range(n):
        returnsentences.append(list(topsentences.keys())[i])
    return returnsentences


if __name__ == "__main__":
    main()
