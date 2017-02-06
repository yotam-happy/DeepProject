import numpy
import sklearn
import sys
import matplotlib.pyplot as plt

#np.seterr(all='ignore')

def sigmoid(x):
    return 1. / (1 + numpy.exp(-x))

def softmax(x):
    e = numpy.exp(x - numpy.max(x))  # prevent overflow
    if e.ndim == 1:
        return e / numpy.sum(e, axis=0)
    else:
        return e / numpy.array([numpy.sum(e, axis=1)]).T  # ndim = 2


class LogisticRegression(object):
    def __init__(self, input, label, n_in, n_out):
        self.x = input
        self.y = label
        self.W = numpy.zeros((n_in, n_out))  # initialize W 0
        self.b = numpy.zeros(n_out)          # initialize bias 0

        # self.params = [self.W, self.b]

    def train(self, lr=0.1, input=None, L2_reg=0.00):
        if input is not None:
            self.x = input

        # p_y_given_x = sigmoid(numpy.dot(self.x, self.W) + self.b)
        p_y_given_x = softmax(numpy.dot(self.x, self.W) + self.b)
        d_y = self.y - p_y_given_x

        self.W += lr * numpy.dot(self.x.T, d_y) - lr * L2_reg * self.W
        self.b += lr * numpy.mean(d_y, axis=0)

        # cost = self.negative_log_likelihood()
        # return cost

    def negative_log_likelihood(self):
        # sigmoid_activation = sigmoid(numpy.dot(self.x, self.W) + self.b)
        sigmoid_activation = softmax(numpy.dot(self.x, self.W) + self.b)

        cross_entropy = - numpy.mean(
            numpy.sum(self.y * numpy.log(sigmoid_activation) +
            (1 - self.y) * numpy.log(1 - sigmoid_activation),
                      axis=1))

        return cross_entropy


    def predict(self, x):
        # return sigmoid(numpy.dot(x, self.W) + self.b)
        return softmax(numpy.dot(x, self.W) + self.b)

def read_matrix(path):
    mat = []
    with open(path, 'r') as f:
        for line in f:
            l = line.split(',')
            l = [float(x) for x in l]
            mat.append(l)
    return numpy.array(mat)

def read_categorical(path, n_classes=10):
    mat = []
    with open(path, 'r') as f:
        for line in f:
            l = line.split(',')
            row = numpy.zeros(n_classes)
            row[int(l[0])-1] = 1.0
            mat.append(row)
    return numpy.array(mat)

def my_predict(classifier, x, y):
    reg_y = classifier.predict(x)
    pred_y = numpy.argmax(reg_y, axis=1)
    correct = 0
    for pred, act in zip(pred_y, y):
        if pred == act:
            correct += 1

    return float(correct) / y.shape[0]


train_x = read_matrix('C:\\Users\\yotamesh\\Google Drive\\Documents\\Second_Degree\\year3_sem1\\kira\\USPS\\tr_X.txt')
train_y = read_categorical('C:\\Users\\yotamesh\\Google Drive\\Documents\\Second_Degree\\year3_sem1\\kira\\USPS\\tr_y.txt')
test_x = read_matrix('C:\\Users\\yotamesh\\Google Drive\\Documents\\Second_Degree\\year3_sem1\\kira\\USPS\\te_X.txt')
test_y = read_categorical('C:\\Users\\yotamesh\\Google Drive\\Documents\\Second_Degree\\year3_sem1\\kira\\USPS\\te_y.txt')
train_actual_y = numpy.argmax(train_y, axis=1)
actual_y = numpy.argmax(test_y, axis=1)

# construct LogisticRegression
classifier = LogisticRegression(input=train_x, label=train_y, n_in=256, n_out=10)
'''
# train
epoch = 0
stop = False
best_precision = 0
no_improvement = 0
train_graph = []
test_graph = []
loss_graph = []
while not stop:
    classifier.train(lr=0.0001, L2_reg=100)
    loss = classifier.negative_log_likelihood()
    loss_graph.append(loss)

    precision = my_predict(classifier, test_x, actual_y)
    train_precision = my_predict(classifier, train_x, train_actual_y)
    if train_precision > best_precision:
        best_precision = train_precision
        no_improvement = 0
    else:
        no_improvement += 1
        if no_improvement > 100:
            stop = True

    train_graph.append(train_precision)
    test_graph.append(precision)
    print 'Training epoch %d, loss is ' % epoch, loss, 'accuracy:', precision, ' train accuracy:', train_precision
    epoch += 1
    if epoch == 5000:
        stop = True


x = [i+1 for i in xrange(len(train_graph))]
plt.plot(x, train_graph, '-b', label='Train')
plt.plot(x, test_graph, '-r', label='Test')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(loc='lower right')
plt.show()
'''
reg = [0,1,2, 3]
train_loss_reg = [0.984, 0.9816, 0.9606, 0.923]
test_loss_reg = [0.921, 0.9216, 0.9208, 0.9]

plt.xticks([0,1,2,3], [0,1,10,100])
plt.plot(reg, train_loss_reg, '-b', label='Train')
plt.plot(reg, test_loss_reg, '-r', label='Test')
plt.ylabel('Accuracy')
plt.xlabel('Regularization')
plt.legend(loc='upper right')
plt.show()
