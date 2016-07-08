# coding:utf-8
import cPickle

import nltk
import numpy as np
import theano as theano
import theano.tensor as T
from utils import *
import lasagne
from lasagne.layers import *
from collections import OrderedDict
import sys
sys.setrecursionlimit(1000)

#theano.config.optimizer = 'fast_compile'
theano.config.floatX = 'float32'


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


class RNNTheano:
    def __init__(self, in_dim, out_dim, batch_size, seq_length_t,em_dim = 620, hidden_dim=1000, v_dim=1000, l_dim=500, grad_clipping=100):
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.batch_size = batch_size
        self.em_dim = em_dim
        self.hidden_dim = hidden_dim
        self.v_dim = v_dim
        self.l_dim = l_dim
        self.grad_clipping = grad_clipping
        self.seq_length_t = seq_length_t
        self.theano = {}

        rng = np.random.RandomState(1234)
        self.Va = theano.shared(np.zeros((v_dim,)).astype("float32"), name="Va")
        self.Wa = theano.shared(np.random.normal(0, 0.001, size=(hidden_dim, v_dim)).astype("float32"),
                                name="Wa")
        self.Ua = theano.shared(np.random.normal(0, 0.001,  size=(hidden_dim * 2, v_dim)).astype("float32"),
                                name="Ua")
        self.Wr = theano.shared(np.random.normal(0, 0.01, size=(em_dim, hidden_dim)).astype("float32"),
                                name="Wr")
        self.Ur = theano.shared(np.random.normal(0, 0.01, size=(hidden_dim, hidden_dim)).astype("float32"),
                                name="Ur")
        self.Cr = theano.shared(np.random.normal(0, 0.01, size=(hidden_dim * 2, hidden_dim)).astype("float32"),
                                name="Cr")
        self.Wz = theano.shared(np.random.normal(0, 0.01, size=(em_dim, hidden_dim)).astype("float32"),
                                name="Wz")
        self.Uz = theano.shared(np.random.normal(0, 0.01, size=(hidden_dim, hidden_dim)).astype("float32"),
                                name="Uz")
        self.Cz = theano.shared(np.random.normal(0, 0.01, size=(hidden_dim * 2, hidden_dim)).astype("float32"),
                                name="Cz")
        self.C = theano.shared(np.random.normal(0, 0.01, size=(hidden_dim * 2, hidden_dim)).astype("float32"),
                               name="C")
        self.U = theano.shared(np.random.normal(0, 0.01, size=(hidden_dim, hidden_dim)).astype("float32"),
                               name="U")
        self.Wo = theano.shared(np.random.normal(0, 0.01, size=(l_dim, out_dim)).astype("float32"), name="Wo")

        self.Uo = theano.shared(np.random.normal(0, 0.01, size=(hidden_dim, l_dim * 2)).astype("float32"),
                                name="Uo")
        self.Co = theano.shared(np.random.normal(0, 0.01, size=(hidden_dim * 2, l_dim * 2)).astype("float32"),
                                name="Co")
        self.Vo = theano.shared(np.random.normal(0, 0.01, size=(em_dim, l_dim * 2)).astype("float32"),
                                name="Vo")
        self.Ws = theano.shared(np.random.normal(0, 0.01, size=(hidden_dim, hidden_dim)).astype("float32"),
                                name="Ws")
        self.Wi = theano.shared(np.random.normal(0, 0.01, size=(em_dim, hidden_dim)).astype("float32"),
                                name="Wi")
        self.eps = theano.shared(np.float32(0.000001),
                                name="eps")
        self.ganma = theano.shared(np.float32(0.95),name = "ganma")
        self.s = theano.shared(np.float32(0), name = 's')
        self.Ein = theano.shared(np.random.normal(0, 0.01, size=(in_dim, em_dim)).astype("float32"),
                                name="Ein")
        self.Eout = theano.shared(np.random.normal(0, 0.01, size=(out_dim, em_dim)).astype("float32"),
                                name="Eout")
        self.r = theano.shared(np.float32(0), name = 'r')
        self.x = T.tensor3('x')
        self.t = T.tensor3('t')
        self.mask = T.matrix('mask')
        self.lr = theano.shared(np.float32(0.95),name = "lr")
        self.params = [self.Va, self.Wa, self.Ua, self.Wr, self.Ur, self.Cr, self.Wz, self.Uz, self.Cz, self.C, self.U,
                       self.Wo, self.Uo, self.Co, self.Vo, self.Ws,self.Wi,self.Ein,self.Eout]


    def model(self):
        def bidirectional(input_var):
            l_emb = lasagne.layers.InputLayer(shape=(self.batch_size, None, self.em_dim), input_var = input_var)
            l_mask = lasagne.layers.InputLayer(shape=(None,None),input_var = self.mask)
            l_forward = lasagne.layers.GRULayer(
                l_emb, self.hidden_dim,mask_input=l_mask,grad_clipping = self.grad_clipping)
            l_backward = lasagne.layers.GRULayer(
                l_emb, self.hidden_dim,mask_input=l_mask,backwards=True,grad_clipping = self.grad_clipping)
            l_concat = lasagne.layers.ConcatLayer([l_forward, l_backward],axis = 2)
            h = lasagne.layers.get_output(l_concat)
            h_1 = lasagne.layers.get_output(l_backward).dimshuffle(1,0,2)[0]
            bi_params = lasagne.layers.get_all_params(l_concat)
            return h,h_1,bi_params


        def decoder_step(yprev, sprev, wr, ur, cr, wz, uz, cz, u, va, wa, ua, uo, vo, wo, co, hnow, c_c, wi, eout):
            def calculate_e(hh, v, w, uu, s_t):
                return T.exp(T.dot(v, (T.tanh(T.dot(s_t, w) + T.dot(hh, uu)))))

            def calculate_c(a,hnow):
                return T.dot(a,hnow)

            def maxout(t):
                #t.reshape(self.l_dim * 2 * self.batch_size,)
                return T.as_tensor_variable([T.max([t[i * 2], t[i * 2 + 1]]) for i in range(self.l_dim)])

            e = T.as_tensor_variable([theano.scan(fn=calculate_e, non_sequences=[va, wa, ua, sprev[i]], sequences=[hnow[i]])[0] for i in range(self.batch_size)])

            a = e / T.sum(e)

            #c = [T.dot(a[i],hnow[i]) for i in range(self.batch_size)]

            c,_ = theano.scan(fn = calculate_c, non_sequences = [], sequences = [a,hnow])

            t_bar = T.dot(sprev, uo) + T.dot(T.dot(yprev,eout), vo) + T.dot(c, co)

            t_t,_ = theano.scan(fn = maxout, non_sequences = [],sequences = t_bar)

            yprev = T.dot(t_t, wo)

            ri = T.nnet.sigmoid(T.dot(T.dot(yprev,eout), wr) + T.dot(sprev, ur) + T.dot(c, cr))
            zi = T.nnet.sigmoid(T.dot(T.dot(yprev,eout), wz) + T.dot(sprev, uz) + T.dot(c, cz))
            si_bar = T.tanh(T.dot(T.dot(yprev,eout),wi) + T.dot((ri * sprev), u) + T.dot(c, c_c))
            sprev = (T.as_tensor_variable(np.ones(shape = (self.batch_size,1)).astype('float32')) - zi) * sprev + (zi * si_bar)


            return yprev, sprev


        def adadelta(ada_param,g):
            ada_param[0] = self.ganma * self.r + (1 - self.ganma) * (g * g)
            v = ((T.sqr(ada_param[1]) + self.eps) / (T.sqr(ada_param[0]) + self.eps)) * g
            ada_param[1] = self.ganma * ada_param[1] + (1 - self.ganma) * (v * v)
            #return T.cast(v, dtype = 'float32')
            return v


        def calculate_cost(predicted_y, target):
            p_y_given_x = T.nnet.softmax(predicted_y.reshape((self.batch_size * self.seq_length_t,self.out_dim)))
            index = T.argmax(target.reshape((self.batch_size * self.seq_length_t,self.out_dim)),axis = 1)
            cost = -T.sum(T.log(p_y_given_x)[T.arange(self.batch_size * self.seq_length_t),index])
            return cost

        emb_x = T.dot(self.x,self.Ein)
        h, h_1, bi_params = bidirectional(emb_x)
        print type(h_1)
        params = self.params + bi_params
        """
        for i in range(batch_size):
            s0 = T.tanh(T.dot(h_1[i], self.Ws))
            [y, s], _ = theano.scan(fn=decoder_step,
                                  non_sequences=[self.Wr, self.Ur, self.Cr, self.Wz, self.Uz, self.Cz, self.U, self.Va,
                                                 self.Wa, self.Ua, self.Uo, self.Vo, self.Wo, self.Co, h[i], self.C, self.Wi, self.Eout],
                                  outputs_info=[np.zeros(shape = (self.out_dim)).astype('float32'), s0], n_steps= self.t[i].shape[0])

            #cost += T.sum((T.nnet.softmax(y) - self.t[i]) ** 2)
            cost += calculate_cost(y,self.t[i])
        """
        s0 = T.tanh(T.dot(h_1,self.Ws))
        [y,s], _ = theano.scan(fn = decoder_step,
                               non_sequences=[self.Wr, self.Ur, self.Cr, self.Wz, self.Uz, self.Cz, self.U, self.Va,
                                                 self.Wa, self.Ua, self.Uo, self.Vo, self.Wo, self.Co, h, self.C, self.Wi, self.Eout],
                               outputs_info=[np.zeros(shape = (self.batch_size,self.out_dim)).astype('float32'), s0], n_steps = self.seq_length_t)
        cost = calculate_cost(y,self.t)
        gparams = T.grad(cost,params)
        print "nice"
        updates = OrderedDict()
        ada_dic = OrderedDict()
        for i in range(len(params)):
            ada_dic[i] = [0,0]

        for i,p,g in zip(range(len(params)),params,gparams):
            v = adadelta(ada_dic[i],g)

            updates[p] = p - v

        compute = theano.function(inputs=[self.x, self.t, self.mask], outputs=cost, updates=updates)

        return compute

def make_mask(data,batch_size,max_length,dim):
    mask = np.zeros(shape = (batch_size,max_length))
    shaped_data = np.zeros(shape = (batch_size,max_length,dim))
    for i,seq in enumerate(data):
        seq_length = len(seq)
        mask[i,:seq_length] = 1
        for j,idx in enumerate(seq):
            vec = np.zeros(dim)
            vec.put(idx,1)
            shaped_data[i][j] = vec

    return mask.astype('float32'),shaped_data

def shuffle_data(data_set):
    np.random.shuffle(data_set)
    return data_set
if __name__ == "__main__":
    ge_tokens = Token()
    en_tokens = Token()
    en_tokens.tokenize("training/train.en")
    ge_tokens.tokenize("training/train.de")
    in_dim = ge_tokens.Vocabsize
    out_dim = en_tokens.Vocabsize
    print in_dim
    print out_dim

    en_training_data = en_tokens.getTrainingData()[:29000]
    ge_training_data = ge_tokens.getTrainingData()[:29000]
    en_max_length = max(len(i) for i in en_training_data)
    ge_max_length = max(len(i) for i in ge_training_data)
    batch_size = 29
    n_epoch = 1000
    print len(en_training_data)
    n_batch =len(en_training_data) / batch_size
    print n_batch
    """RNNTheano takes in_dim and out_dim as input"""
    # rnn_en = RNNTheano(en_tokens.Vocabsize,ge_tokens.Vocabsize)
    rnn_en = RNNTheano(in_dim, out_dim,batch_size,en_max_length)


    model = rnn_en.model()

    print "compile success"
    for epoch in range(n_epoch):
        for i in range(n_batch):
            x_data = ge_training_data[i * batch_size:(i+1) * batch_size]
            t_data = en_training_data[i * batch_size:(i+1) * batch_size]
            input_mask,shaped_x_data = make_mask(x_data,batch_size,ge_max_length,in_dim)
            _,shaped_t_data = make_mask(t_data,batch_size,en_max_length,out_dim)
            shaped_x_data = shaped_x_data.astype('float32')
            shaped_t_data = shaped_t_data.astype('float32')
            cost = model(shaped_x_data,shaped_t_data,input_mask)
            print cost
        ge_training_data = shuffle_data(x_data)
        en_training_data = shuffle_data(t_data)

        f = open("models/model_%s.dump" % epoch)
        cPickle.dump(rnn_en,f,protocol=cPickle.HIGHEST_PROTOCOL)
        f.close()




