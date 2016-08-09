# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'deepesa_ui.ui'
#
# Created by: PyQt4 UI code generator 4.11.4
#
# WARNING! All changes made in this file will be lost!
import os

from PyQt4 import QtCore, QtGui

import RNNModel
from DbWrapper import WikipediaDbWrapper
from KnockoutModel import KnockoutModel
from WikilinksIterator import WikilinksNewIterator
from WikilinksStatistics import WikilinksStatistics
from Word2vecLoader import Word2vecLoader

try:
    _fromUtf8 = QtCore.QString.fromUtf8
except AttributeError:
    def _fromUtf8(s):
        return s

try:
    _encoding = QtGui.QApplication.UnicodeUTF8
    def _translate(context, text, disambig):
        return QtGui.QApplication.translate(context, text, disambig, _encoding)
except AttributeError:
    def _translate(context, text, disambig):
        return QtGui.QApplication.translate(context, text, disambig)

class Ui_MainWindow(object):
    def __init__(self, w2v_stats_dic = None):
        self.model = None
        self.db = None
        self.itr = None
        self.path = "C:\\repo\\DeepProject"
        self.password = 'rockon123'
        self.w2v_stats_dic = w2v_stats_dic
        if w2v_stats_dic is not None:
            self.itr = w2v_stats_dic['iter'].get_wlink()
        if(not os.path.isdir(self.path)):
            self.path = "C:\\Users\\Noam\\Documents\\GitHub\\DeepProject"
            self.password = 'ncTech#1'

    '''
    all functional definitions off UI. connect() connects buttons to functions
    '''
    def setupUi(self, MainWindow):
        MainWindow.setObjectName(_fromUtf8("MainWindow"))
        MainWindow.resize(863, 627)
        self.centralwidget = QtGui.QWidget(MainWindow)
        self.centralwidget.setObjectName(_fromUtf8("centralwidget"))
        self.gridLayout = QtGui.QGridLayout(self.centralwidget)
        self.gridLayout.setObjectName(_fromUtf8("gridLayout"))
        self.verticalLayout_2 = QtGui.QVBoxLayout()
        self.verticalLayout_2.setObjectName(_fromUtf8("verticalLayout_2"))
        self.rcontext_LBL = QtGui.QLabel(self.centralwidget)
        self.rcontext_LBL.setObjectName(_fromUtf8("rcontext_LBL"))
        self.verticalLayout_2.addWidget(self.rcontext_LBL)

        self.lcontext_TXT = QtGui.QLineEdit(self.centralwidget)
        self.lcontext_TXT.setObjectName(_fromUtf8("lcontext_TXT"))

        self.verticalLayout_2.addWidget(self.lcontext_TXT)
        self.label_2 = QtGui.QLabel(self.centralwidget)
        self.label_2.setObjectName(_fromUtf8("label_2"))
        self.verticalLayout_2.addWidget(self.label_2)

        self.word_TXT = QtGui.QLineEdit(self.centralwidget)
        self.word_TXT.setObjectName(_fromUtf8("word_TXT"))

        self.verticalLayout_2.addWidget(self.word_TXT)
        self.lcontext_LBL = QtGui.QLabel(self.centralwidget)
        self.lcontext_LBL.setObjectName(_fromUtf8("lcontext_LBL"))
        self.verticalLayout_2.addWidget(self.lcontext_LBL)

        self.rcontext_TXT = QtGui.QLineEdit(self.centralwidget)
        self.rcontext_TXT.setObjectName(_fromUtf8("rcontext_TXT"))

        # Buttons
        self.verticalLayout_2.addWidget(self.rcontext_TXT)
        self.label_4 = QtGui.QLabel(self.centralwidget)
        self.label_4.setObjectName(_fromUtf8("label_4"))
        self.verticalLayout_2.addWidget(self.label_4)
        self.log_TXT = QtGui.QTextBrowser(self.centralwidget)
        self.log_TXT.setObjectName(_fromUtf8("log_TXT"))
        self.verticalLayout_2.addWidget(self.log_TXT)
        self.gridLayout.addLayout(self.verticalLayout_2, 0, 0, 1, 1)
        self.verticalLayout = QtGui.QVBoxLayout()
        self.verticalLayout.setObjectName(_fromUtf8("verticalLayout"))
        self.babelfy_BTN = QtGui.QPushButton(self.centralwidget)
        self.babelfy_BTN.setMinimumSize(QtCore.QSize(0, 50))
        self.babelfy_BTN.setObjectName(_fromUtf8("babelfy_BTN"))
        self.verticalLayout.addWidget(self.babelfy_BTN)

        self.run_BTN = QtGui.QPushButton(self.centralwidget)
        self.run_BTN.setMinimumSize(QtCore.QSize(0, 50))
        self.run_BTN.setObjectName(_fromUtf8("run_BTN"))
        self.run_BTN.clicked.connect(self._run_wsd)
        self.verticalLayout.addWidget(self.run_BTN)

        self.rnd_BTN = QtGui.QPushButton(self.centralwidget)
        self.rnd_BTN.setMinimumSize(QtCore.QSize(0, 50))
        self.rnd_BTN.setObjectName(_fromUtf8("rnd_BTN"))
        self.rnd_BTN.clicked.connect(self._get_wikilink)
        self.verticalLayout.addWidget(self.rnd_BTN)

        self.exit_BTN = QtGui.QPushButton(self.centralwidget)
        self.exit_BTN.setMinimumSize(QtCore.QSize(0, 50))
        self.exit_BTN.setObjectName(_fromUtf8("exit_BTN"))
        self.exit_BTN.clicked.connect( QtCore.QCoreApplication.instance().quit)

        self.verticalLayout.addWidget(self.exit_BTN)
        self.gridLayout.addLayout(self.verticalLayout, 0, 2, 1, 1)
        MainWindow.setCentralWidget(self.centralwidget)

        # Menu
        self.menubar = QtGui.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 863, 21))
        self.menubar.setObjectName(_fromUtf8("menubar"))
        self.menuLoad = QtGui.QMenu(self.menubar)
        self.menuLoad.setObjectName(_fromUtf8("menuLoad"))

        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtGui.QStatusBar(MainWindow)
        self.statusbar.setObjectName(_fromUtf8("statusbar"))
        MainWindow.setStatusBar(self.statusbar)

        self.actionLoad = QtGui.QAction(MainWindow)
        self.actionLoad.setObjectName(_fromUtf8("actionLoad"))
        self.actionLoad.triggered.connect(self._load_model)

        self.menuLoad.addAction(self.actionLoad)
        self.menubar.addAction(self.menuLoad.menuAction())

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    '''
    UI functions
    '''
    def _load_model(self):
        # selecting the model
        fname = QtGui.QFileDialog.getOpenFileName()
        f = open(fname, 'r')

        # loading the model
        if self.w2v_stats_dic is None:
            self.w2v_stats_dic = self.loadW2VandStats()
            self.itr =  self.w2v_stats_dic['iter'].get_wlink()

        rnn = RNNModel.RNNPairwiseModel(self.w2v_stats_dic['w2v'])
        model_file = rnn.loadModel((fname.__str__()).split('.model')[0])
        self.model = KnockoutModel(model_file, self.w2v_stats_dic['stats'])
        self.log_TXT.setText(str(fname)+' model was loaded....')
        self.db = WikipediaDbWrapper(user='root', password=self.password, database='wikiprep-esa-en20151002')

    def _run_wsd(self):
        rcontext = str(self.rcontext_TXT.text())
        lcontext = str(self.lcontext_TXT.text())
        mention = str(self.word_TXT.text())
        if self.model is not None:
             self.log_TXT.setText('we have a model')
             output = self.model.predict2(lcontext,mention,rcontext,self.db)
             self.log_TXT.setText(str(output))
        else:
            self.log_TXT.setText('no model was loaded...')

    def _get_wikilink(self):
        if self.w2v_stats_dic is None:
            self.log_TXT.setText('No iterator. Please load model....')
            return
        wlink = self.iter.next()
        self.rcontext_TXT.setText(' '.join(wlink['right_context']))
        self.lcontext_TXT.setText(' '.join(wlink['left_context']))
        self.word_TXT.setText(wlink['word'])

    def retranslateUi(self, MainWindow):
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow", None))
        self.rcontext_LBL.setText(_translate("MainWindow", "Left Context", None))
        self.label_2.setText(_translate("MainWindow", "Word", None))
        self.lcontext_LBL.setText(_translate("MainWindow", "Right Context", None))
        self.label_4.setText(_translate("MainWindow", "Log", None))
        self.babelfy_BTN.setText(_translate("MainWindow", "Babelfy", None))
        self.run_BTN.setText(_translate("MainWindow", "Run", None))
        self.rnd_BTN.setText(_translate("MainWindow", "Random txt", None))
        self.exit_BTN.setText(_translate("MainWindow", "Exit", None))
        self.menuLoad.setTitle(_translate("MainWindow", "Model", None))
        self.actionLoad.setText(_translate("MainWindow", "Load", None))

    def _closeUI(self):
        QtCore.QCoreApplication.instance().quit

    def loadW2VandStats(self):
        path = "/home/yotam/pythonWorkspace/deepProject"
        print "Loading w2v and stats..."
        if(not os.path.isdir(path)):
            path = "C:\\Users\\Noam\\Documents\\GitHub\\DeepProject"

        train_stats = WikilinksStatistics(None, load_from_file_path=path+"/data/wikilinks/small/wikilinks.stats")
        iter_eval = WikilinksNewIterator(path+"/data/wikilinks/small_evaluation",
                                 mention_filter=train_stats.getGoodMentionsToDisambiguate(f=10))
        w2v = Word2vecLoader(wordsFilePath=path+"/data/word2vec/dim300vecs",
                             conceptsFilePath=path+"/data/word2vec/dim300context_vecs")
        wD = train_stats.contextDictionary
        cD = train_stats.conceptCounts
        w2v.loadEmbeddings(wordDict=wD, conceptDict=cD)
        print 'Done!'
        return {'w2v': w2v, 'stats': train_stats, 'iter':iter_eval}

if __name__ == "__main__":
    import sys
    app = QtGui.QApplication(sys.argv)
    MainWindow = QtGui.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())




