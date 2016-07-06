from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import SGD

class VanillaNN:
    """
    This model is a baseline NN with simple architecture. It has an prediction
    and train function and is part of the knockoutmodel system.
    the input of the model is assumed to be 3 embedding vectors with shape (300,) each
    for the mention, the candidate sense and the context embedding sense . the output should be a simple (2,)
    binary vector, specifying which candidate is more accurate
    """

    def __init__(self, w2v):
        self._w2v = w2v

        # model initialization
        # Multi layer percepatron -2 hidden layers with 64 fully connected neurons
        self.model = Sequential()
        self.model.add(Dense( 64 ,input_dim = 900 , init = 'uniform' ))
        self.model.add(Activation('tanh'))
        self.model.add(Dense( 64 , init = 'uniform' ))
        self.model.add(Activation('tanh'))
        self.model.add(Dense(2, init = 'uniform'))
        self.model.add(Activation('softmax'))

        # defining solver and compile
        sgd = SGD(lr=0.1, decay=1e-6,momentum=0.9)
        self.model.compile(loss='binary_crossentropy',optimizer='sgd')

    def train(self):
        return None

