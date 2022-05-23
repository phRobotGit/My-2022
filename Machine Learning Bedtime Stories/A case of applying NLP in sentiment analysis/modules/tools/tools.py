import numpy as np
from difflib import SequenceMatcher
import copy
# functions to compute TF-IDF
def computeTF(wordDict, doc):
    # first input has each word of the document with relative number of times the word is in that document
    # second input is words in the document
    # init dictionary
    tfDict = {}
    corpusCount = len(doc)
    for word, count in wordDict.items():
        tfDict[word] = count/float(corpusCount)
    return(tfDict)
def computeIDF(Doc_count_word, N):
    # first input is word in all documents with how many documents contain that word
    # second input is the number of document
    # init dictionary
    idfDict = {}
    idfDict = dict.fromkeys(Doc_count_word.keys(), 0)
    for word in Doc_count_word:
        idfDict[word] = np.log10(N / float(Doc_count_word[word]))
    return(idfDict)
def computeTFIDF(tfBow, idfs):
    # just take ttfBow and idfs and deliver td*idf for each word
    # init dictionary
    tfidf = {}
    for word, val in tfBow.items():
        tfidf[word] = val*idfs[word]
    return(tfidf)
    
# function to map word to sentiments
def map_word_to_sent(Doc_count_word, positives, negatives, similarity_check = True):
    Doc_word_LM = dict.fromkeys(Doc_count_word.keys(), 0)
    for word in Doc_word_LM:
        if word in positives:
            Doc_word_LM[word] = 1
        elif word in negatives:
            Doc_word_LM[word] = -1
        # Otherwise (and if option activated - see below) check similarities
        else:
            # similarity check: https://kite.com/python/docs/difflib.SequenceMatcher.ratio
            # T is the total number of elements in both sequences, and M is the number of matches, ratio is: 2.0*M / T
            if similarity_check:
                # Select most similar word
                this_sim = 0
                for j in positives:
                    score_sim = SequenceMatcher(None, j, word).ratio()
                    if score_sim > this_sim : this_sim = copy.deepcopy(score_sim)
                Doc_word_LM[word] = this_sim
                # If similarity is less than 0.95 then check negatives as well
                if this_sim < 0.95:
                    for j in negatives:
                        score_sim = SequenceMatcher(None, j, word).ratio()
                        if score_sim > this_sim :
                            this_sim = copy.deepcopy(score_sim)
                            Doc_word_LM[word] = - this_sim
    return Doc_word_LM