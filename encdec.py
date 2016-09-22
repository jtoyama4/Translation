# coding:utf-8
#import sys
import numpy
import numpy as np
#from argparse import ArgumentParser
from chainer import Chain, Variable, cuda, functions, links, optimizer, optimizers, serializers,FunctionSet
from chainer.functions import EmbedID,Linear,transpose_sequence,lstm,tanh,where,softmax_cross_entropy
from chainer.optimizers import *
import nltk
import os
#os.environ["CHAINER_TYPE_CHECK"] = "0" #type_checkしない
#import util.generators as gens
#from util.functions import trace, fill_batch
#from util.vocabulary import Vocabulary

class Token:
    def __init__(self, tokens=None):
        self.tokens = tokens
        self.Vocabsize = 0
        self.index = None

    def tokenize(self, file_place):
        f = open(file_place, "r")
        text = f.readlines()
        f.close()

        tokens = []
        vocabs = []
        for line in text:
            line = line.decode("utf8")
            tokens.append(nltk.word_tokenize(line))
            vocabs.extend(nltk.word_tokenize(line))
        self.tokens = tokens
        vocabs = list(set(vocabs))
        index = dict([(w, i) for i, w in enumerate(vocabs)])
        self.Vocabsize = len(vocabs)
        self.index = index

    def getTrainingData(self):
        training = []
        for token in self.tokens:
            sentence = [self.index[word] for word in token]
            training.append(sentence)
        return training

ge_tokens = Token()
en_tokens = Token()
ge_tokens.tokenize("training/train.de")
en_tokens.tokenize("training/train.en")
in_dim = ge_tokens.Vocabsize
out_dim = en_tokens.Vocabsize


ge_training_data = ge_tokens.getTrainingData()[:2900]
en_training_data = en_tokens.getTrainingData()[:2900]
tgt_index = dict([(v,k) for k,v in en_tokens.index.items()])
#ge_max_length = max(len(i) for i in ge_training_data)
#en_max_length = max(len(i) for i in en_training_data)


SRC_VOCAB_SIZE = in_dim
SRC_EMBED_SIZE = 100
HIDDEN_SIZE = 150
TRG_EMBED_SIZE = 100
TRG_VOCAB_SIZE = out_dim + 1

model = FunctionSet(
    w_xi = EmbedID(SRC_VOCAB_SIZE, SRC_EMBED_SIZE),
  w_ip = Linear(SRC_EMBED_SIZE, 4 * HIDDEN_SIZE),
  w_pp = Linear(HIDDEN_SIZE, 4 * HIDDEN_SIZE),
  w_pq = Linear(HIDDEN_SIZE, 4 * HIDDEN_SIZE),
  w_yq = EmbedID(TRG_VOCAB_SIZE, 4 * HIDDEN_SIZE),
  w_qq = Linear(HIDDEN_SIZE, 4 * HIDDEN_SIZE), 
  w_qj = Linear(HIDDEN_SIZE, TRG_EMBED_SIZE),
  w_jy = Linear(TRG_EMBED_SIZE, TRG_VOCAB_SIZE),
)  
# src_sentence: 翻訳したい単語列 e.g. ['彼', 'は', '走る']
# trg_sentence: 正解の翻訳を表す単語列 e.g. ['he', 'runs']
# training: 学習か予測か。デコーダの挙動に影響する。
def forward(src_sentence, trg_sentence, model, training):

  # 単語IDへの変換（自分で適当に実装する）
  # 正解の翻訳には終端記号を追加しておく。
  #src_sentence = [convert_to_your_src_id(word) for word in src_sentence]
  #trg_sentence = [convert_to_your_trg_id(word) for wprd in trg_sentence] + [END_OF_SENTENCE]
  trg_sentence.append(out_dim)
  # LSTM内部状態の初期値
  c = Variable(np.zeros((1, HIDDEN_SIZE), dtype=np.float32))
  p = Variable(np.zeros((1, HIDDEN_SIZE), dtype=np.float32))
  # エンコーダ
  for word in reversed(src_sentence):
    x = Variable(np.array([[word]], dtype=np.int32))
    i = tanh(model.w_xi(x))
    c, p = lstm(c, model.w_ip(i) + model.w_pp(p))

  # エンコーダ -> デコーダ
  c, q = lstm(c, model.w_pq(p))

  # デコーダ
  if training:
    # 学習時はyとして正解の翻訳を使い、forwardの結果として累積損失を返す。
    accum_loss = 0
    for word in trg_sentence:
      j = tanh(model.w_qj(q))
      y = model.w_jy(j)
      t = Variable(np.array([word], dtype=np.int32))
      accum_loss += softmax_cross_entropy(y, t)
      c, q = lstm(c, model.w_yq(t) +  model.w_qq(q))
    return accum_loss
  else:
    # 予測時には翻訳器が生成したyを次回の入力に使い、forwardの結果として生成された単語列を返す。
    # yの中で最大の確率を持つ単語を選択していくが、softmaxを取る必要はない。
    hyp_sentence = []
    while len(hyp_sentence) < 100: # 100単語以上は生成しないようにする
      j = tanh(model.w_qj(q))
      y = model.w_jy(j)
      word = y.data.argmax(1)[0]
      if word == out_dim:
        break # 終端記号が生成されたので終了
      y = Variable(np.array([word], dtype=np.int32))
      hyp_sentence.append(tgt_index[word])
      c, q = lstm(c, model.w_yq(y) + model.w_qq(q))
    return hyp_sentence

def padding(batch):
    length = [len(x) for x in batch]
    max_len = max(length)
    for x in batch:
        x_len = len(x)
        for i in range(max_len - x_len):
            x.append(-1)
    return batch

def set_seq(x_batch,y_batch):
    x_y = [(x_batch[i],y_batch[i]) for i in range(10)]
    x_y.sort(key=lambda x:len(x[0]),reverse = True)
    x_batch = [numpy.asarray(x_y[i][0],dtype = np.int32) for i in range(10)]
    y_batch = [numpy.asarray(x_y[i][1],dtype = np.int32) for i in range(10)]
    return x_batch,y_batch

def train(source_set,target_set, model):
  opt = ada_delta() # 確率的勾配法を使用
  opt.setup(model) # 学習器の初期化
  count = 0
  for epoch in range(20):
      for x,y in zip(source_set,target_set):
          count += 1
          opt.zero_grads()
          accum_loss = forward(x,y,model,training = True) # 損失の計算
          print accum_loss.data
          accum_loss.backward() # 誤差逆伝播
          opt.clip_grads(10) # 大きすぎる勾配を抑制
          opt.update() # パラメータの更新
          if count % 100 == 0:
              print count
              sentence = forward(x,y,model,training = False)
              print sentence

train(ge_training_data,en_training_data,model)
