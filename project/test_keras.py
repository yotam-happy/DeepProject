from keras.models import Sequential
from keras.layers import Dense, Activation
import random
import numpy as np

def generateClusters(mu_sigma, s):
    X = []
    Y = []
    for i in xrange(0, s):
        cluster = random.randrange(len(mu_sigma))
        x = np.array([random.gauss(mu,sigma) for (mu,sigma) in mu_sigma[cluster]])
        y = np.array([(1 if k == cluster else 0) for k in xrange(0, len(mu_sigma))])
        X.append(x)
        Y.append(y)
    return np.array(X), np.array(Y)


# generates train and test for [n_clusters] clusters of gaussian shape in dimension [dim]
def getTrainTest(dim, n_clusters):
    # these are the cluster mu and sigma per each dimension per each cluster
    mu_sigma = [[(random.uniform(0,10), random.uniform(1,4)) for d in xrange(0, dim)] for i in xrange(0, n_clusters)]

    train = generateClusters(mu_sigma, 3000 * n_clusters)
    test = generateClusters(mu_sigma, 100 * n_clusters)
    return train, test

model = Sequential()
model.add(Dense(output_dim=15, input_dim=20))
model.add(Activation("relu"))
model.add(Dense(output_dim=10))
model.add(Activation("softmax"))

model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
train, test = getTrainTest(20, 10)
model.fit(train[0], train[1], nb_epoch=5, batch_size=32)
loss_and_metrics = model.evaluate(test[0], test[1], batch_size=32)
print "evaluatio!!"
print "-----------"
print loss_and_metrics