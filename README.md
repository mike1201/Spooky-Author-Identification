#  Text classificaion with 3 classes : Predict the author.
## https://www.kaggle.com/c/spooky-author-identification
## Used Model : “CNN, LSTM, FASTTEXT, XG-Boost”
### CNN, LSTM, XG-Boost : Bad Result   
### USE FASTTEXT : Good Result  

### 1. Preprocessing.
Given the characteristic of the data, instead of using the characteristic of a word, we should seek to capture the unique expressions used by writers. Through a lot of analysis, we asked for the best way to deal with the data.  

#### 1-1. To preserve punctuation marks, separate punctuation from words.
> In general, punctuation marks are normalized and removed. But removing them makes it difficult to capture writers' unique expressions. Thus, preprocessing includes preserving punctuation marks. In order to do so, separate them from words. 
> e.g. I love you. -> I love you .  


#### 1-2. Cutting long sentence.
> Too long documents are cut  

#### 1-3. Removing lower frequency words ( <= 2 or 3)
> To avoid overfitting, remove lower frequency words.  

#### 1-4. Removing Stopwords.
> Stopwords are generally words that appear commonly at high frequency in a corpus. They don't actually contribute much to the learning or predictive process as a learning model.

#### 1-5. Stemming or Lemmatization.
We can reduce many different variations of similar words into a single term. However, there is one flaw with stemming. The process involves quite a crude heuristic in chopping off the ends of words, since it reduces a particular word to a base form that human can recognize. So use lemmatization instead. Lemmatizing the dataset aims to reduce words based on an actual dictionary and therefore will not chop off words into stemmed forms that do not carry any lexical meaning.





### 2. FASTTEXT

> Model architecture of FASTTEXT for a sentence with N n-grams features x1, x2, .. xN. The features are embedded randomly and averaged to form the hidden variable. ( Target word is replaced with classification item. )

#### 2-1. Embeddings
> If we use glove or google vector, punctuation mark is not meaningful. Actually, we used those vectors, and we had a difficult time classifying the data. So we used FASTTEXT to preserve the unique expressions used by writers.  

#### 2-2. n-grams.
> Various ngrams methods were used, and 2gram had the best performance. Also, to keep the unique expressions, we also tried “jump(?)” ( he is real --) he-real )

#### 2-3. AveragePooling 1D
After embedding, the average was calculated by each word embedding variable when preprocessing one data, that is, sentence.
ex)  size : sen_length, Embedding_size  --)  1, Embedding_size 


#### 2-3-1. Features of the sentences.
> We also tried to add a feature to sentences. Specifically, we got word frequency from each author's corpus and added the sum of them as variables.
> ex) size : 1, Embedding_size --) 1, Embedding_size + 3
