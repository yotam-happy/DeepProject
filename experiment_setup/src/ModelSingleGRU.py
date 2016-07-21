from RNNModel import *


class ModelSingleGRU(RNNPairwiseModel):
    def __init__(self, w2v, context_window_sz = 20):
        self._w2v = w2v
        self._batch_context_X = []
        self._batch_candidates_X = []
        self._batchY = []
        self._context_window_sz = context_window_sz
        self._train_loss = []

        # model initialization
        self._batch_size = 512
        print 'creating single GRU model...'
        context_input = Input(shape=(self._context_window_sz*2,self._w2v.embeddingSize), name='context_input')
        candidates_input = Input(shape=(self._w2v.embeddingSize * 2,), name='candidates_input')
        gru_unit = GRU(self._w2v.embeddingSize)(context_input)
        x = merge([gru_unit,candidates_input], mode='concat')
        x = Dense(300, activation='relu')(x)
        x = Dense(50, activation='relu')(x)
        out = Dense(2, activation='softmax', name='main_output')(x)

        model = Model(input=[context_input,candidates_input], output=[out])
        model.compile(optimizer='adagrad', loss='binary_crossentropy')
        self.model = model

    def _trainXY(self,left_X, right_X, candidates_X,Y):
        context_X = np.concatenate([left_X,right_X],0)
        # print 'left X and right_X ', left_X.shape,' ',right_X.shape
        # print 'context shape ',context_X.shape
        self._batch_context_X.append(context_X )
        self._batch_candidates_X.append(candidates_X)
        self._batchY.append(Y)

        if len(self._batchY) >= self._batch_size:
            # pushes numeric data into batch vector
            batch_context_X = np.array(self._batch_context_X)
            # print 'batch_contex_X ',batch_context_X.shape
            batch_candidates_X = np.array(self._batch_candidates_X)
            # print 'batch_can_X ',batch_candidates_X.shape
            batchY = np.array(self._batchY)
            # print 'batch_Y ',batchY.shape
            # training on batch is specifically good for cases were data doesn't fit into memory
            loss = self.model.train_on_batch({'context_input':batch_context_X,
                                              'candidates_input':batch_candidates_X},
                                             batchY)
            self._train_loss.append(loss)
            print 'Done batch. Size of batch - ', batchY.shape, '; loss: ', loss
            # print self.model.metrics_names

            self._batch_context_X = []
            self._batch_candidates_X = []
            self._batchY = []

    def predict(self, wikilink, candidate1, candidate2):
        vecs = self._2vec(wikilink, candidate1, candidate2)
        if not isinstance(vecs, tuple):
            return vecs
        (left_X, right_X, candidates_X) = vecs
        context_X = np.concatenate([left_X,right_X],0)
        # print 'concate completed'
        context_X = context_X.reshape(1,context_X.shape[0],context_X.shape[1])
        candidates_X = candidates_X.reshape(1,candidates_X.shape[0])
        Y = self.model.predict({'context_input':context_X,
                                'candidates_input':candidates_X},
                               batch_size=1)
        return candidate1 if Y[0][0] > Y[0][1] else candidate2
