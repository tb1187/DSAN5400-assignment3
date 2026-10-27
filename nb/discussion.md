# Assignment 3

**DSAN-5400: Computational Linguistics**
*Tyler Blue*

---

## Problem 1

### Naive Bayes Comparison

#### How is it different from yours?
There are a few differences in implementation of the Naive Bayes model. First, when constructing the likelihoods, the solution took lidstone smoothing into account in the denominator as alpha * (|V| + 1) whereas my solution had lidstone smoothing in the denominator as alpha * |V|. This causes the rows to not sum to 1. Additionaly, the test function calls np.log(likelihoods) inside the for text in df['text'] loop. This can be computed outside the loop once instead of every time. 

#### What could you have done differently?
There is a big assumnption that the integer labels are going to be coded 0, 1, and my solution would break immediately if they are not. There are also a few instances where counting loops are written twice. Tokenization could have been improved beyong splitting on white space as well. 

#### What do you think I could have done better?
The accuracy and the F1 score could have been printed along with the confusion-matrix plot.

### pytest Output
```
(/home/c18ty/DSAN_5400/.venv) c18ty@TylerLaptop:~/DSAN_5400/week_4/assignment3/nb$ pytest tests/
===================================================================== test session starts =====================================================================
platform linux -- Python 3.14.7, pytest-9.1.1, pluggy-1.6.0
rootdir: /home/c18ty/DSAN_5400/week_4/assignment3/nb
configfile: pytest.ini
plugins: anyio-4.14.2
collected 3 items                                                                                                                                             

tests/test_naive_bayes.py ...                                                                                                                           [100%]

====================================================================== 3 passed in 0.89s ======================================================================
```

## Problem 2

### What are the universal properties of a tokenizer that you should include in your interface?
From my actual implementation:
- tokenize(text) turns a string into a list of tokens, and every subclass must implement it
- normalize(tokens) lowercases and strips punctuation, shared by all tokenizers
- __call__ runs one after the other

### Using each index, what document ids are returned for the query "system?"
The document ids returned for every index were ['3', '1127', '2552', '3319', '5653', '8772', '8935', '14398', '16515', '16638']. Even though the same documents were retured, the term counts differ greatly between the tokenizers. The whitespace, nltk, and ngram tokenizers had 1686, 1747, and 4894 terms respectively.  

### Using each index, what document ids are returned for the query "compatibility"?
The document ids returned for every index were ['8772', '14398']. Like the first query, all of the same documents were returned across the methods. This leads me to believe that the word "compatibility" occurs commonly and consistently across documents. 

### Using each index, what document ids are returned for the query terms "system" and "compatability"?
The document ids returned for every index were ['8772', '14398']. The same reasoning as the first two queries can be applied here. 