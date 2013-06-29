import unittest
from mock import *
import numpy as np
import cPickle as Pickle
import os

import pycassa

from elm import ELMData, numpyArrayToCassandra, cassandraToNumpyArray, myLeaveOneOut, ELM


# Defining the function for emulating Cassandra batch inserts
def mock_batchinsert(key, dictionary):
    theType = dictionary.keys()[0]
    fileHandle = open('/home/ymiche/Desktop/'+key+'-'+theType+'.txt', 'wb')
    fileHandle.write(dictionary[theType])
    fileHandle.close()

def mock_columnfamilyget(key, columns):
    data = dict()
    fileHandle = open('/home/ymiche/Desktop/'+key+'-values.txt', 'r')
    data['values'] = fileHandle.read()
    fileHandle.close()
    fileHandle = open('/home/ymiche/Desktop/'+key+'-shape.txt', 'r')
    data['shape'] = fileHandle.read()
    fileHandle.close()
    return data
    
    


class ELMClassifiertest(unittest.TestCase):
    def setUp(self):
        self.mockConnectionPool = Mock(spec=pycassa.pool.ConnectionPool)
        self.minimizationType = 'FP'
        self.numberNeurons = 10
        np.random.seed(1)

    @patch('elm.pycassa.columnfamily.ColumnFamily')
    def test_train_wisconsin1(self, mockFunction):
        print 'Yes, I am running, be patient, Human.'
        returnedMock = Mock(spec=pycassa.ColumnFamily)
        returnedMock.get = mock_columnfamilyget
        returnedMock.batch = Mock()
        mockbatchinsert = Mock()
        mockbatchinsert.insert = mock_batchinsert
        mockbatch = Mock()
        mockbatch.__enter__ = Mock(return_value=mockbatchinsert)
        mockbatch.__exit__ = Mock(return_value=False)
        mockFunction.return_value = returnedMock
        returnedMock.batch.return_value = mockbatch
        self.myELM = ELM(self.minimizationType, self.mockConnectionPool, self.numberNeurons)
        inputData = np.loadtxt('./test_train_input_wisconsin1.txt').reshape((379, 30))
        outputData = np.loadtxt('./test_train_output_wisconsin1.txt').reshape((379, 1))
        self.myELM.ELMTrainingData = ELMData(inputData, outputData, [], [])
        self.myELM.storeTrainingDataToCassandra()
        self.myELM.performTraining()
        self.myELM.trained = True

        print 'Best LOO result retained: %s' % self.myELM.bestClassificationRateLOO

        inputTestData = np.loadtxt('./test_test_input_wisconsin1.txt').reshape((190, 30))
        outputTestData = np.loadtxt('./test_test_output_wisconsin1.txt').reshape((190, 1))
        testData = ELMData(inputTestData, outputTestData, [], [])
        confidence, verdict = self.myELM.verdict(testData)
        print 'Test error: %s' % np.mean(verdict == outputTestData)

        listOfErrors = [item for item in xrange(len(verdict)) if verdict[item]!=outputTestData[item]]
        print confidence[listOfErrors]





if __name__ == '__main__':
    unittest.main()