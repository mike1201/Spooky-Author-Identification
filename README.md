#  Predict the excerpts by three authors.

## Used Model : “CNN, LSTM, FASTTEXT, XG-Boost”

### 1. CNN, LSTM, XG-Boost : Bad Result

### 2. FASTTEXT

### 2-1. Preprocessing.
> Given the characteristic of the data, instead of using the characteristic of a word, we should seek to capture the unique expressions used by writers. Through a lot of analysis, we asked for the best way to deal with the data.

### 2-1-1. To preserve punctuation marks, separate punctuation from words.
> In general, the punctuation mark is noramalized and removed. But Removing a punctuation mark makes it difficult to catch the writer`s unique expression. So preprocessing was carried out by preserving the punctuation mark. So separate them from words.
> e.g. Don`t worry --> Don ` t worry

### 2-1-2. Cutting long sentence.
> Too long documents are cut.

2-1-3. Remove lower frequency words ( <= 2 or 3)
To avoid overfitting, remove lower frequency words.

2-1-3. Embeddings.
If we use glove or google vector, punctuation mark is not meaningful. Actually, we used those vectors, and we had a difficult time classifying the data. So we use FASTTEXT to preserve the unique expressions used by writers.

2-1-3-1. n-grams.
Various ngrams methods were used, and 2gram had the best performance. Also, to keep the unique expressions, we also tried “jump(?)” ( he is real --) he-real )

2-2. AveragePooling 1D
After embedding, the average was calculated by each word`s embedding variable when preprocessing one data, that is, sentence.
ex> sentence :  [ sen_length, Embedding_size ] --) [ 1, Embedding_size ]

2-2-1. Feature of the sentences.
We also tried to add a feature to the sentence. Specifically, we get word frequency from each author`s corpus. and add the sum of them as variables.
: [1, Embedding_size] --) [1, Embedding_size + 3 ]

2-3. Fully Connected layer.


