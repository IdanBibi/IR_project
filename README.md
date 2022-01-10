# IR_project

Search Engine on the whole English Wikipedia.

The followings are descriptions of the modules provided in this repository:

## <b>inverted_index_gcp<b>
This module was given to us in assignment 3.
It has various fields which contain information about the total number of a term's frequency, the amount of documents each term appears in and the posting list for each term.
It is capable of reading and loading an already existing index from local or remote storage. 

## <b>Backend<b>
This module provides functions that helps us to search for the correct document for a specific query.
Some of the functions are:
***tfidf_func*** which calculates the tfidf scores for documents and returns the sorted.
***get_docs_binary*** which calculates the amount of words from the query that appears in the given index.
***tokenize*** which turns the query into tokens.
***expand_query*** which expands the query by adding synonyms.
***sim*** calculates similarity between different tokens in the query and filtering the words that are not similar to others.
***read_posting_list*** which reads a posting list of a single word.

## <b>search_fronted<b>
The main module where the search function exists.

### <ins><b>search<b><ins>
This is the main search function which results with the our best map@40 score of 0.519. <br />
This search calculates the relevent documents by this algorithm:
  if the length of the query tokens is 2 or less we use get_docs_binary on the title and sort the results.
  else we send the query to sim function which changes the query due to the similarities between the tokens. After the output from sim function we checked whether the query changed or not. If the query changed we use get_docs_binary on the new query on the title and sort the results. Else we use expand the query and use get_docs_binary on the body  and sort the results.

### <ins><b>search_body<b><ins> <br />
Searching through the body of the articles using the tfidf_func function from the Backend module.<br /> <br />

### <ins><b>search_title<b><ins> <br />
Searching through the title of the articles using the get_docs_binary function from the Backend module. <br /> <br />

### <ins><b>search_anchor<b><ins> <br />
Searching through the acnhor text of the articles using the get_docs_binary function from the Backend module.
  
### <ins><b>get_pagerank<b><ins> <br />
Pulling the pre-calculated pagerank value from the storage. <br /> <br />

### <ins><b>get_pageview<b><ins> <br />
Pulling the pre-calculated pageviews value from the storage. <br /> <br />

