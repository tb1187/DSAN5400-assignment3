# Assignment 3

**DSAN-5400: Computational Linguistics**
*Tyler Blue*

---

## Problem 1

### Naive Bayes Comparison

#### How is it different from yours?
There are a few differences in implementation of the Naive Bayes model. First, when constructing the likelihoods, my approach did not take lidstone smoothing into account in the denominator whereas the solution had lidstone smoothing in the denominator. This causes the rows to not sum to 1. Additionaly, the test function calls np.log(likelihoods) inside the for text in df['text'] loop. This can be computed outside the loop once instead of every time. 

#### What could you have done differently?
There is a big assumnption that the integer labels are going to be coded 0, 1, and my solution would break immediately if they are not. There are also a few instances where counting loops are written twice. Tokenization could have been improved beyong splitting on white space as well. 

#### What do you think I could have done better?
The accuracy and the F1 score could have been printed along with the confusion-matrix plot.

### pytest Output
(/home/c18ty/DSAN_5400/.venv) c18ty@TylerLaptop:~/DSAN_5400/week_4/assignment3/nb$ pytest tests/
===================================================================== test session starts =====================================================================
platform linux -- Python 3.14.7, pytest-9.1.1, pluggy-1.6.0
rootdir: /home/c18ty/DSAN_5400/week_4/assignment3/nb
configfile: pytest.ini
plugins: anyio-4.14.2
collected 3 items                                                                                                                                             

tests/test_naive_bayes.py ...                                                                                                                           [100%]

====================================================================== 3 passed in 0.89s ======================================================================

## Problem 2

### What are the universal properties of a tokenizer that you should include in your interface?

### Using each index, what document ids are returned for the query "system?"

### Using each index, what document ids are returned for the query "compatibility"?

### Using each index, what document ids are returned for the query terms "system" and "compatability"?