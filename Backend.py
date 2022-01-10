import pandas as pd
import numpy as np
import json
import re
import nltk
from nltk.corpus import stopwords
from gensim.models import KeyedVectors
from inverted_index_gcp import *

nltk.download('stopwords')

# get the files
inverted_text = InvertedIndex.read_index("/home/idanbib/IR_Project_Data/postings_gcp_text",
                                         'index')
inverted_title = InvertedIndex.read_index("/home/idanbib/IR_Project_Data/postings_gcp_title", 'index')
inverted_anchor = InvertedIndex.read_index("/home/idanbib/IR_Project_Data/postings_gcp_anchor", 'index')

loc_text = "/home/idanbib/IR_Project_Data/postings_gcp_text/"
loc_title = "/home/idanbib/IR_Project_Data/postings_gcp_title/"
loc_anchor = "/home/idanbib/IR_Project_Data/postings_gcp_anchor/"

id_title_dict = pd.read_pickle("/home/idanbib/IR_Project_Data/id_title_dict.pickle")
with open("/home/idanbib/IR_Project_Data/doc_lengths.json") as f:
    id_lengths_dict = json.load(f)
with open("/home/idanbib/IR_Project_Data/page_rank_data.json") as f:
    pr = json.load(f)
pw = pd.read_pickle("//home/idanbib/IR_Project_Data/pageviews.pkl")

model = KeyedVectors.load_word2vec_format("/home/idanbib/IR_Project_Data/wiki-news-300d-1M.vec")

## FUNCTIONS
def get_docs_binary(query, index, file_name, expand=False):
    '''
    Returns ALL (not just top 100) search results that contain A QUERY WORD
        IN THE TITLE OR ANCHOR of articles, ordered in descending order of the NUMBER OF
        QUERY WORDS that appear in the title. For example, a document with a
        title that matches two of the query words will be ranked before a
        document with a title that matches only one query term.
    Args:
        query: the query we need to search on
        index: which index we want to check
        file_name: the path of the index
        expand: expand the query or not

    Returns: sorted list of the ids that query appears in

    '''
    ids_and_words = {}
    for term in np.unique(tokenize(query, expand)):
        if term in index.df.keys():
            list_of_doc = read_posting_list(index, term, file_name)
            for doc_id, _ in list_of_doc:
                if doc_id in ids_and_words:
                    ids_and_words[doc_id] += 1
                else:
                    ids_and_words[doc_id] = 1
    return sorted(ids_and_words, key=ids_and_words.get, reverse=True)


def read_posting_list(inverted, w, file_name):
    '''
    Read a posting list of a single word
    Args:
        inverted: the inverted index
        w: word
        file_name: the path of the index

    Returns: a posting list of the word

    '''
    posting_list = []
    with closing(MultiFileReader()) as reader:
        if w in inverted.df:
            locs = inverted.posting_locs[w]
            locs = [(file_name + lo[0], lo[1]) for lo in locs]
            b = reader.read(locs, inverted.df[w] * TUPLE_SIZE)
            for i in range(inverted.df[w]):
                doc_id = int.from_bytes(b[i * TUPLE_SIZE:i * TUPLE_SIZE + 4], 'big')
                tf = int.from_bytes(b[i * TUPLE_SIZE + 4:(i + 1) * TUPLE_SIZE], 'big')
                posting_list.append((doc_id, tf))
    return posting_list


def tokenize(text, expand=False):
    '''
    Tokenize the query into tokens.
    Args:
        text: the query
        expand: expand the query or not

    Returns: list of tokens

    '''
    english_stopwords = frozenset(stopwords.words('english'))
    corpus_stopwords = ["category", "references", "also", "external", "links",
                        "may", "first", "see", "history", "people", "one", "two",
                        "part", "thumb", "including", "second", "following",
                        "many", "however", "would", "became", 'make']
    RE_WORD = re.compile(r"""[\#\@\w](['\-]?\w){2,24}""", re.UNICODE)
    all_stopwords = english_stopwords.union(corpus_stopwords)

    list_of_tokens = [token.group() for token in RE_WORD.finditer(text.lower()) if
                      token.group() not in all_stopwords]
    if expand:
        list_of_tokens = expand_query(list_of_tokens)
    return list_of_tokens


def tfidf_func(query, inverted, file_name, expand=False):
    '''
    Returns a sorted list of documents id's according to tf-idf score
    Args:
        query: the query we need to search on
        inverted: which index we want to check
        file_name: the path of the index
        expand: expand the query or not

    Returns: Returns a sorted list of documents id's according to tf-idf score

    '''
    query_tfidf = {}
    doc_tfidf = defaultdict(list)
    numerator = {}
    cos = {}
    query_tokens = tokenize(query, expand)
    for token in query_tokens:
        post_list = read_posting_list(inverted, token, file_name)
        idf = np.log10(len(id_lengths_dict) / (inverted.df[token]))
        query_tfidf[token] = query_tokens.count(token) * idf / (len(query_tokens))
        for doc_id, freq in post_list:
            if doc_id == 0:
                continue
            doc_tfidf[token].append((doc_id, freq * idf / id_lengths_dict[str(doc_id)]))

    for key, val in doc_tfidf.items():
        for doc_id, tfidf in val:
            if doc_id in numerator:
                numerator[doc_id] += tfidf * query_tfidf[key]
            else:
                numerator[doc_id] = tfidf * query_tfidf[key]

    for key, val in numerator.items():
        cos[key] = val / (id_lengths_dict[str(key)] * len(query_tokens))

    return sorted(cos, key=cos.get, reverse=True)


def expand_query(query_tokens):
    '''
    Expands the query
    Args:
        query_tokens: token of a given query

    Returns: a new list of expanded tokens

    '''
    new_tokens = []
    for tok in query_tokens:
        if tok in model:
            sim = model.most_similar(tok, topn=5)
            for word in sim:
                new_tokens.append(word[0])
        new_tokens.append(tok)
    return new_tokens

def sim(query_tokens):
    '''
    Calculates similarity between different tokens in the query and filtering the words that are not similar to others.
    Args:
        query_tokens: token of a given query

    Returns: query.

    '''
    similarity = []
    final = []
    for i, w1 in enumerate(query_tokens):
        for j, w2 in enumerate(query_tokens):
            if i != j and w1 in model and w2 in model:
                similarity.append((w1, w2, model.similarity(w1, w2)))
    for i in similarity:
        if i[2] >= 0.39:
            final.append(i[0])
            final.append(i[1])
    if final:
        return True, ' '.join(final)
    return False, ' '.join(query_tokens)