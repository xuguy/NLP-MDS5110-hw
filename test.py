# !wget -O quora.zip -qq --no-check-certificate "https://drive.google.com/uc?export=download&id=1ERtxpdWOgGQ3HOigqAMHTJjmOE_tWvoF"
# !unzip -o quora.zip

# use wget to download and unzip manually
# downloaded file are renamed to quora.zip
# then unzip quora.zip we get train.csv
'''
# download dataset, we have already download one so no need

import wget
url = "https://drive.google.com/uc?export=download&id=1ERtxpdWOgGQ3HOigqAMHTJjmOE_tWvoF"
data = wget.download(url)
'''



import nltk
nltk.download('punkt') # will be download to Roaming/nltk_data ~22MB
import time
import math
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from IPython.display import clear_output
%matplotlib inline
np.random.seed(42)

import pandas as pd
from nltk.tokenize import word_tokenize
from tqdm import tqdm

#%%
quora_data = pd.read_csv('train.csv')

# there might be duplicate questions of both q1 and q2
# quora_data.question1.shape = (404290,)
# quora_data.question2.unique().shape = (299175,)
quora_data.question1 = quora_data.question1.replace(np.nan, '', regex=True)
quora_data.question2 = quora_data.question2.replace(np.nan, '', regex=True)

texts = list(pd.concat([quora_data.question1, quora_data.question2]).unique())

# extract the first 50000 entries of data
texts = texts[:50000] # Accelerated operation
print(len(texts))

tokenized_texts = [word_tokenize(text.lower()) for text in tqdm(texts)]
'''
texts[1]: 'What is the story of Kohinoor (Koh-i-Noor) Diamond?'

tokenized_texts[1]:
['what',
 'is',
 'the',
 'story',
 'of',
 'kohinoor',
 '(',
 'koh-i-noor',
 ')',
 'diamond',
 '?']
'''

assert len(tokenized_texts) == len(texts)
assert isinstance(tokenized_texts[0], list)
assert isinstance(tokenized_texts[0][0], str)

#%%
from collections import Counter

MIN_COUNT = 5

# 将二维列表 tokenized_texts 展开为一个一维的token流，便于统计所有token的词频，二维列表就是需要逐层展开
words_counter = Counter(token for tokens in tokenized_texts for token in tokens)
word2index = {
    '<unk>': 0
}

# .most_common() 按照出现频次又高到低排列
'''
words_counter.most_common():

[('?', 52593),
 ('the', 23345),
 ('what', 20102),
 ('is', 16787),
 ('how', 13372),
 ('i', 13012),
 ('a', 12622),
 ('to', 12361),
 ('in', 12214),
 ('do', 10274),
 ('of', 9740),
 ('are', 9163),
 ('and', 8193),
'''
for word, count in words_counter.most_common():
    # 词频小于5的直接忽略
    if count < MIN_COUNT:
        break
    # 每加一个词进word2index，长度加以，以此来assign index
    word2index[word] = len(word2index)

index2word = [word for word, _ in sorted(word2index.items(), key=lambda x: x[1])]

print('Vocabulary size:', len(word2index))
print('Tokens count:', sum(len(tokens) for tokens in tokenized_texts))
print('Unknown tokens appeared:', sum(1 for tokens in tokenized_texts for token in tokens if token not in word2index))
print('Most freq words:', index2word[1:21])
'''
Vocabulary size: 7226
Tokens count: 623563
Unknown tokens appeared: 35607
Most freq words: ['?', 'the', 'what', 'is', 'how', 'i', 'a', 'to', 'in', 'do', 'of', 'are', 'and', 'can', 'for', ',', 'you', 'why', 'it', 'best']
'''

#%%
def build_contexts(tokenized_texts, window_size):
    contexts = []
    for tokens in tokenized_texts:
        for i in range(len(tokens)):
            central_word = tokens[i]
            context = [tokens[i + delta] for delta in range(-window_size, window_size + 1)
                       if delta != 0 and i + delta >= 0 and i + delta < len(tokens)]

            contexts.append((central_word, context))

    return contexts

contexts = build_contexts(tokenized_texts, window_size=2)
# %%
contexts[:5]
'''
[('what', ['is', 'the']),
 ('is', ['what', 'the', 'step']),
 ('the', ['what', 'is', 'step', 'by']),
 ('step', ['is', 'the', 'by', 'step']),
 ('by', ['the', 'step', 'step', 'guide'])]
'''

# %%
# convert words into indices
# .get(str, value_to_return_if_str_does_not_exits)
contexts = [(word2index.get(central_word, 0), [word2index.get(word, 0) for word in context])
            for central_word, context in contexts]

# make batch:
window_size = 2
batch_size = 32

'''
# 测试代码：
# 把开头和结尾两个词剔除，因为他们不包含完整的window_size为2的contexts
central_words_tmp = np.array([word for word, context in contexts if len(context) == 2 * window_size and word != 0])

contexts_tmp = np.array([context for word, context in contexts if len(context) == 2 * window_size and word != 0])

# math.ceil() 向上取整
# 13238
batches_count = int(math.ceil(len(contexts_tmp) / batch_size))
# (423591,) 个 contexts 各自的编号
indices = np.arange(len(contexts_tmp))
# 随机打乱，并覆盖原array
np.random.shuffle(indices)

batch_indices = indices[0:32]

'''


def make_cbow_batches_iter(contexts, window_size, batch_size):

    central_words = np.array([word for word, context in contexts if len(context) == 2 * window_size and word != 0])
    contexts = np.array([context for word, context in contexts if len(context) == 2 * window_size and word != 0])


    batches_count = int(math.ceil(len(contexts) / batch_size))
    print(f'central_words: {central_words.shape}, contexts: {contexts.shape}\n')
    # batch_size=32，那么就需要 12380个batch以后才完整地过了一遍数据集
    print('Initializing batches generator with {} batches per epoch'.format(batches_count))

    indices = np.arange(len(contexts))
    np.random.shuffle(indices)

    for i in range(batches_count):
      # 最后一个batch可能不够batch_size个数据
      batch_begin, batch_end = i * batch_size, min((i + 1) * batch_size, len(contexts))
      batch_indices = indices[batch_begin: batch_end]

      # ------------------
      # Write your implementation here.
      contexts_batch = torch.LongTensor(contexts[batch_indices])
      central_word_batch = torch.LongTensor(central_words[batch_indices])

      batch = {'tokens': contexts_batch, 'labels': central_word_batch}
      # avoid producing all batch at once, this is to improv effi
      yield batch
      # ------------------

# verification:

window_size = 2
batch_size = 32

# next()这行代码每运行一次，make_cbow里面的循环就会往下走一步
batch = next(make_cbow_batches_iter(contexts, window_size=window_size, batch_size=batch_size))

assert isinstance(batch, dict)
assert 'labels' in batch and 'tokens' in batch

assert isinstance(batch['tokens'], torch.LongTensor)
assert isinstance(batch['labels'], torch.LongTensor)

assert batch['tokens'].shape == (batch_size, 2 * window_size)
assert batch['labels'].shape == (batch_size,)

#%%
# implement CBoW:
class CBoWModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super().__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.out_layer = nn.Linear(embedding_dim, vocab_size)

    def forward(self, inputs):
        # ------------------
        # Write your implementation here.
        embedded = self.embeddings(inputs)  # 形状: (batch_size, 2*window_size, embedding_dim)
        
        # 沿上下文维度（dim=1）取平均
        mean_embedded = torch.mean(embedded, dim=1)  # 形状: (batch_size, embedding_dim)
        
        # 通过线性层输出logits（未归一化的概率）
        outputs = self.out_layer(mean_embedded)  # 形状: (batch_size, vocab_size)
        
        return outputs
        # ------------------

# cuda #1 ====
# initialize model for test:
model = CBoWModel(vocab_size=len(word2index), embedding_dim=32)#.cuda()

# cuda #2 ====
# outputs = model(batch['tokens'].cuda())
# batch['tokens'] = contexts
'''
example:
array([['what', 'is', 'step', 'by'],
       ['is', 'the', 'by', 'step'],
       ['the', 'step', 'step', 'guide'],
       ...,
       ['phone', 'storage', 'my', 'galaxy'],
       ['storage', 'on', 'galaxy', '5s'],
       ['on', 'my', '5s', '?']], dtype='<U41')
'''
outputs = model(batch['tokens'])

# assert isinstance(outputs, torch.cuda.FloatTensor)
assert isinstance(outputs, torch.FloatTensor)
assert outputs.shape == (batch_size, len(word2index))

# ==== training loop without GPU =====

# Here are the hyperparameters you can adjust
embedding_dim = 32
learning_rate = 0.001
epoch_num = 5
batch_size = 128

# Initialization Model
# len(word2index) = 7226
model = CBoWModel(len(word2index),embedding_dim)
# Getting model to GPU
# cuda #3 ===
# model.cuda()
# loss function
criterion = nn.CrossEntropyLoss()
# use Adam optimizer
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

loss_every_nsteps = 3000 # report loss every 3000 steps
total_loss = 0
start_time = time.time()
global_step = 0

for ep in range(epoch_num):
  for step, batch in enumerate(make_cbow_batches_iter(contexts, window_size=2, batch_size=batch_size)):
      global_step += 1

      # ------------------
      # Write your implementation here.
      # cuda #4 ====
      inputs = batch['tokens']#.cuda()  # contexts
      labels = batch['labels']#.cuda()  # central_words

      # forward
      outputs = model(inputs)  # shape: (batch_size, vocab_size)

      # cal loss
      loss = criterion(outputs, labels)

      # bp
      optimizer.zero_grad()  # zero out grad every new batch
      loss.backward()        # bp
      optimizer.step()       # update weights
      # ------------------

      total_loss += loss.item()

      if global_step != 0 and global_step % loss_every_nsteps == 0:
          print("Epoch = {}, Step = {}, Avg Loss = {:.4f}, Time = {:.2f}s".format(ep, step, total_loss / loss_every_nsteps,
                                                                      time.time() - start_time))
          total_loss = 0
          start_time = time.time()

# =============== finish training ================
#%%
embeddings = model.embeddings.weight.data.cpu().numpy()
# embeddings shape: (7226, 32)
# does not matter if there is a .cpu() if you are using cpu
# if you dont convert tensor from cuda to cpu if your torch version is gpu, error:
# TypeError: can't convert cuda:0 device type tensor to numpy. Use Tensor.cpu() to copy the tensor to host memory first.

# testing training result:
from sklearn.metrics.pairwise import cosine_similarity

def most_similar(embeddings, index2word, word2index, word):
    word_emb = embeddings[word2index[word]]

    similarities = cosine_similarity([word_emb], embeddings)[0]
    top10 = np.argsort(similarities)[-10:]

    return [index2word[index] for index in reversed(top10)]

most_similar(embeddings, index2word, word2index, 'my')


# Visualization:
import bokeh.models as bm, bokeh.plotting as pl
from bokeh.io import output_notebook

from sklearn.manifold import TSNE
from sklearn.preprocessing import scale


def draw_vectors(x, y, radius=10, alpha=0.25, color='blue',
                 width=600, height=400, show=True, **kwargs):
    """ draws an interactive plot for data points with auxilirary info on hover """
    output_notebook()

    if isinstance(color, str):
        color = [color] * len(x)
    data_source = bm.ColumnDataSource({ 'x' : x, 'y' : y, 'color': color, **kwargs })

    fig = pl.figure(active_scroll='wheel_zoom', width=width, height=height)
    fig.scatter('x', 'y', size=radius, color='color', alpha=alpha, source=data_source)

    fig.add_tools(bm.HoverTool(tooltips=[(key, "@" + key) for key in kwargs.keys()]))
    if show:
        pl.show(fig)
    return fig


def get_tsne_projection(word_vectors):
    tsne = TSNE(n_components=2, verbose=1)
    return scale(tsne.fit_transform(word_vectors))


def visualize_embeddings(embeddings, index2word, word_count):
    word_vectors = embeddings[1: word_count + 1]
    words = index2word[1: word_count + 1]

    word_tsne = get_tsne_projection(word_vectors)
    draw_vectors(word_tsne[:, 0], word_tsne[:, 1], color='blue', token=words)


visualize_embeddings(embeddings, index2word, 100)

# %%
# there is a scipy version conflict here: the scipy version used here is version==1.13, which is over up-to-date, downgrade it to version == 1.12 will work, remeber to reboot kernel
# import gensim.downloader as api
'''
gensim basic data structure source code:
https://github.com/piskvorky/gensim/blob/develop/gensim/models/keyedvectors.py
'''

def load_word2vec():
    """ Load GloVe Twitter Vectors
        Return:
            wv_from_bin: Pre-trained embeddings with 25 dimensions for 1.2M vocabulary.
    """
    import gensim.downloader as api
    # glove-twitter-25 (vocab_size, embedding_dim)=(1.2M, 25)
    wv_from_bin = api.load("glove-twitter-25")
    vocab = list(wv_from_bin.key_to_index.keys())  # Updated for Gensim 4.x
    print("Loaded vocab size %i" % len(vocab))
    return wv_from_bin


wv_from_bin = load_word2vec()
dir(wv_from_bin)
import pickle
file_name = 'wv_from_bin.pkl'

# # dump
# with open(file_name, 'wb') as f:
#     pickle.dump(wv_from_bin, f)

# re-load
with open(file_name, 'rb') as f:
    wv_from_bin = pickle.load(f)

words_tmp = list(wv_from_bin.key_to_index.keys())

# ===== reducing dimensionality =====
import numpy as np

def get_matrix_of_vectors(wv_from_bin, required_words=['barrels', 'bpd', 'ecuador', 'energy', 'industry', 'kuwait', 'oil', 'output', 'petroleum', 'venezuela']):
    """ Put the word2vec vectors into a matrix M.
        Param:
            wv_from_bin: KeyedVectors object; the 1.2 million word2vec vectors loaded from file
        Return:
            M: numpy matrix shape (num words, 300) containing the vectors
            word2Ind: dictionary mapping each word to its row number in M
    """
    import random
    words = list(wv_from_bin.key_to_index.keys())
    print("Shuffling words ...")
    random.shuffle(words)
    words = words[:10000]
    print("Putting %i words into word2Ind and matrix M..." % len(words))
    word2Ind = {}
    M = []
    curInd = 0
    for w in words:
        try:
            # wv_from_bin.word_vec(w)将w转换为向量
            M.append(wv_from_bin.word_vec(w))
            # 把w映射为indices
            word2Ind[w] = curInd
            curInd += 1
        except KeyError:
            continue
    for w in required_words:
        try:
            M.append(wv_from_bin.word_vec(w))
            word2Ind[w] = curInd
            curInd += 1
        except KeyError:
            continue
    M = np.stack(M)
    print("Done.")
    return M, word2Ind

from sklearn.decomposition import TruncatedSVD
def reduce_to_k_dim(M, k=2):
    """ Reduce a co-occurrence count matrix of dimensionality (num_corpus_words, num_corpus_words)
        to a matrix of dimensionality (num_corpus_words, k) using TruncatedSVD.

        Params:
            M (numpy matrix of shape (num_corpus_words, num_corpus_words)): co-occurrence matrix of word counts
            k (int): embedding size of each word after dimension reduction
        Return:
            M_reduced (numpy matrix of shape (num_corpus_words, k)): matrix of k-dimensional word embeddings.
    """
    n_iters = 10  # Number of iterations for SVD
    print("Running Truncated SVD over %i words..." % (M.shape[0]))
    
    # ------------------
    # Write your implementation here.
    svd = TruncatedSVD(n_components=k, n_iter=n_iters, random_state=0)  # truncated svd
    M_reduced = svd.fit_transform(M)  # dimensionality reduction

    # ------------------
    print("Done.")
    return M_reduced

import matplotlib.pyplot as plt

def plot_embeddings(M_reduced, word2Ind, words):
    """ Plot in a scatterplot the embeddings of the words specified in the list "words".
        Include a label next to each point.

        Params:
            M_reduced (numpy matrix of shape (number of unique words in the corpus, k)): matrix of k-dimensional word embeddings
            word2Ind (dict): dictionary that maps word to indices for matrix M
            words (list of strings): words whose embeddings we want to visualize
    """
    plt.figure(figsize=(10, 10))
    for word in words:
        if word in word2Ind:
            idx = word2Ind[word]
            x, y = M_reduced[idx, 0], M_reduced[idx, 1]
            plt.scatter(x, y, marker='o', color='blue')
            plt.text(x + 0.02, y + 0.02, word, fontsize=9)
        else:
            print(f"Word '{word}' not found in word2Ind dictionary.")

    plt.title("Word Embeddings Visualization")
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    plt.grid(True)
    plt.show()

M, word2Ind = get_matrix_of_vectors(wv_from_bin)
M_reduced = reduce_to_k_dim(M, k=2)

words = ['barrels', 'bpd', 'ecuador', 'energy', 'industry', 'kuwait', 'oil', 'output', 'petroleum', 'venezuela']
plot_embeddings(M_reduced, word2Ind, words)

'''
问题：
1. get_matrix_ 中要求return Return:
            M: numpy matrix shape (num words, 300) containing the vectors 中的300是为什么？明明wv_from_bin的向量只有25d

2. 为什么要把reqiured_words append到M里？万一M里面包含required_words的embedded vec，有什么影响？
部分自回答：把required_words加到M里是因为我们想要画出这些words在2维平面的坐标，因此需要把它们加入M，然后一起进行truncatedSVD。猜测：如果M里面已经包含了required_words，因为vocab_size=10000比较大，所以加入10个就算有重复也对我们的的visualization没有多大影响？

3. Find a polysemous word (for example, "leaves" or "scoop") such that the top-10 most similar words (according to cosine similarity) contains related words from both meanings. For example, "leaves" has both "turns" and "ground" in the top 10, and "scoop" has both "buckets" and "pops" 是什么意思？turns和ground有什么联系？随便输入一个词，他的top10里面都有不一样意思的词啊？

''' 

# find polysemous words:
wv_from_bin.key_to_index.keys()