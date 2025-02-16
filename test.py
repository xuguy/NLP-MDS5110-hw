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