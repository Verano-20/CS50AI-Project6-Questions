# CS50AI-Project6-Questions
An AI to answer questions.

This AI will take a corpus of documents and a query given by the user (a question in English). It will determine the document(s) and sentence(s) most relevant to the query. 

The documents' relevances are ranked by tf-idf (term frequency - inverse document frequency), and the sentences' relevances are ranked by a combination of idf and a query term density measure.

Usage examples:

$ python questions.py corpus  
Query: What are the types of supervised learning?  
Types of supervised learning algorithms include Active learning , classification and regression.

$ python questions.py corpus  
Query: When was Python 3.0 released?  
Python 3.0 was released on 3 December 2008.

$ python questions.py corpus  
Query: How do neurons connect in a neural network?  
Neurons of one layer connect only to neurons of the immediately preceding and immediately following layers.