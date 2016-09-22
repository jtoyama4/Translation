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
a = numpy.array([1,2,3])
print max(a)

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
#ge_max_length = max(len(i) for i in ge_training_data)
#en_max_length = max(len(i) for i in en_training_data)


SRC_VOCAB_SIZE = in_dim
SRC_EMBED_SIZE = 100
HIDDEN_SIZE = 150
TRG_EMBED_SIZE = 100
TRG_VOCAB_SIZE = out_dim + 1
BATCH_SIZE = 10

model = FunctionSet(
    w_xi = EmbedID(SRC_VOCAB_SIZE, SRC_EMBED_SIZE,ignore_label = -1),
  w_ip = Linear(SRC_EMBED_SIZE, 4 * HIDDEN_SIZE),
  w_pp = Linear(HIDDEN_SIZE, 4 * HIDDEN_SIZE),
  w_pq = Linear(HIDDEN_SIZE, 4 * HIDDEN_SIZE),
  w_yq = EmbedID(TRG_VOCAB_SIZE, 4 * HIDDEN_SIZE),
  w_qq = Linear(HIDDEN_SIZE, 4 * HIDDEN_SIZE), 
  w_qj = Linear(HIDDEN_SIZE, TRG_EMBED_SIZE),
  w_jy = Linear(TRG_EMBED_SIZE, TRG_VOCAB_SIZE),
)  

def forward(src_sentence, trg_sentence, model, training=True):
    end = out_dim
    # 単語IDへの変換（自分で適当に実装する）
    # 正解の翻訳には終端記号を追加しておく。
    #src_sentence = [convert_to_your_src_id(word) for word in src_sentence]
    #trg_sentence = [convert_to_your_trg_id(word) for wprd in trg_sentence]
    
    # LSTM内部状態の初期値
    c_prev = Variable(np.zeros((10, HIDDEN_SIZE), dtype=np.float32))
    p_prev = Variable(np.zeros((10, HIDDEN_SIZE), dtype=np.float32))
    i = Variable(np.zeros((10, SRC_EMBED_SIZE), dtype=np.float32))
    # エンコーダ
    for word in reversed(src_sentence):
        word = np.array(word,dtype=np.int32)
        word = word.reshape(10,1)
        x = Variable(np.array(word, dtype=np.int32))
        i = model.w_xi(word)
        c, p = lstm(c_prev, model.w_ip(i) + model.w_pp(p_prev))
        enable = np.asarray([[(x_i != -1) for i in range(HIDDEN_SIZE)] for x_i in x.data.reshape(10,)])
        enable = Variable(enable)
        _c = []
        _p = []
        for i in range(BATCH_SIZE):
            _ = where(enable[i], c[i], c_prev[i])
            _c.append(_.data)
        for i in range(BATCH_SIZE):
            _ = where(enable[i], p[i].data, p_prev[i].data)
            _p.append(_.data)
        c_prev = Variable(np.asarray(_c,dtype = np.float32))
        p_prev = Variable(np.asarray(_p,dtype = np.float32))
    # エンコーダ -> デコーダ
    c, q = lstm(c, model.w_pq(p))
    # デコーダ
    if training:
        # 学習時はyとして正解の翻訳を使い、forwardの結果として累積損失を返す。
        accum_loss = 0
        for word in trg_sentence:
            j = tanh(model.w_qj(q))
            y = model.w_jy(j)
            #y = functions.reshape(y,(1,1,TRG_VOCAB_SIZE))
            #_t = np.zeros(TRG_VOCAB_SIZE,dtype = np.int32)
            #_t[word] = 1
            t = np.asarray(word, dtype= np.int32)
            #t = t.reshape(1,BATCH_SIZE)
            t = Variable(t)
            accum_loss += softmax_cross_entropy(y,t)
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
            if word == END_OF_SENTENCE:
                break # 終端記号が生成されたので終了
            hyp_sentence.append(convert_to_your_trg_str(word))
            c, q = lstm(c, model.w_yq(y), model.w_qq(q))
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
  source_set = np.transpose(source_set)
  target_set = np.transpose(target_set)
  opt.zero_grads() # 勾配の初期化
  accum_loss = forward(source_set,target_set,model,training = True) # 損失の計算
  print accum_loss.data
  accum_loss.backward() # 誤差逆伝播
  opt.clip_grads(10) # 大きすぎる勾配を抑制
  opt.update() # パラメータの更新


opt = SGD()
opt.setup(model)

for epoch in range(20):
    for i in range(len(ge_training_data) / 10):
        #x,y = set_seq(ge_training_data[i * 10: i * 10 + 10],en_training_data[i * 10: i * 10 + 10])
        x = padding(ge_training_data[i * 10: i * 10 + 10])
        y = padding(en_training_data[i * 10: i * 10 + 10])
        train(x,y,model)
