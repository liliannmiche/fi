import unittest
from mock import *
import numpy as np
import cPickle as Pickle
import os
import random

import pycassa
from elm import ELM
from elm import ELMData, numpyArrayToCassandra, cassandraToNumpyArray, myLeaveOneOut, ELM


class ELMDataTest(unittest.TestCase):

    def setUp(self):
        # Testing the initialization
        self.inputCorrect = np.random.randn(100,4)
        self.outputCorrect = np.random.randn(100,1)  

        # Wrong data type
        self.inputIncorrectDataType = ['something','is','wrong','here']
        self.outputIncorrectDataType = ['and','here','as','well']
        # Wrong dimensionality of ndarrays
        self.inputIncorrectDimensionality = np.random.randn(100)
        self.outputIncorrectDimensionality = np.random.randn(100)
        self.outputIncorrectDimensionality2 = np.random.randn(100,2)
        # Non matching sizes
        self.inputIncorrectSize = np.random.randn(100,4)
        self.outputIncorrectSize = np.random.randn(99,1)

    def test_IncorrectArguments(self):
        # Checking data type (numpy ndarray)
        self.assertRaises(Exception, ELMData, self.inputIncorrectDataType, self.outputCorrect)
        self.assertRaises(Exception, ELMData, self.inputCorrect, self.outputIncorrectDataType)
        # Checking that ndarrays are 2d, nothing else
        self.assertRaises(Exception, ELMData, self.inputIncorrectDimensionality, self.outputCorrect)
        self.assertRaises(Exception, ELMData, self.inputCorrect, self.outputIncorrectDimensionality)
        # Checking that input and output have the right (matching) sizes
        self.assertRaises(Exception, ELMData, self.inputIncorrectSize, self.outputIncorrectSize)
        # Checking that output is a Nx1 array (yet still 2d!)
        self.assertRaises(Exception, ELMData, self.inputCorrect, self.outputIncorrectDimensionality2)


    def test_normalizeData(self):
        # Create a ELMData object and test on random (normal) data
        myELMData = ELMData(self.inputCorrect, self.outputCorrect, [], [])
        myELMData.normalizeData()
        # And on zero data it must fail because std is zero for some variables
        myELMData = ELMData(np.zeros((10,4)),np.random.randn(10,1),[],[])
        self.assertRaises(Exception, myELMData.normalizeData)


# Defining the function for emulating Cassandra batch inserts
def mock_batchinsert(key, dictionary):
    theType = dictionary.keys()[0]
    fileHandle = open('./'+key+'-'+theType+'.txt', 'wb')
    fileHandle.write(dictionary[theType])
    fileHandle.close()

def mock_columnfamilyget(key, columns):
    data = dict()
    fileHandle = open('./'+key+'-values.txt', 'r')
    data['values'] = fileHandle.read()
    fileHandle.close()
    fileHandle = open('./'+key+'-shape.txt', 'r')
    data['shape'] = fileHandle.read()
    fileHandle.close()
    return data
    
def mock_columnfamilyget_range(columns, read_consistency_level):
    mockcolumnfamily = Mock(spec=pycassa.ColumnFamily)
    mockcolumnfamily.batch = Mock()
    mockbatchinsert = Mock()
    mockbatchinsert.insert = mock_batchinsert
    mockbatch = Mock()
    mockbatch.__enter__ = Mock(return_value=mockbatchinsert)
    mockbatch.__exit__ = Mock(return_value=False)
    mockcolumnfamily.batch.return_value = mockbatch
    bogusData = np.random.randn(1000, 4)
    bogusData[:, 3] = np.sign(bogusData[:,3])
    numpyArrayToCassandra(bogusData, mockcolumnfamily, 'test_getrange')
    data = dict()
    fileHandle = open('./test_getrange-values.txt', 'r')
    data['values'] = fileHandle.read()
    fileHandle.close()
    fileHandle = open('./test_getrange-shape.txt', 'r')
    data['shape'] = fileHandle.read()
    fileHandle.close()
    myArray = Pickle.loads(data['values']).reshape(Pickle.loads(data['shape']))
    mylist = []
    for row in xrange(myArray.shape[0]):
        mylist.append((hex(random.getrandbits(128))[2:-1], {'NN_dist': myArray[row, 0], 'NNd_dist': myArray[row, 1], 'NNd_rank':myArray[row, 2], 'class': myArray[row, 3]}))
    myiter = iter(mylist)
    return myiter
    
    

    

class globalFunctionsTest(unittest.TestCase):

    # This misses tests with erroneous input, ATM

    def setUp(self):
        self.keyCorrect = 'myfilename'
        self.numpyArrayCorrect = np.random.randn(10,4)
      
    # def testIncorrectArguments(self):
    #     self.assertRaises(Exception, numpyArrayToCassandra, )
    #     self.assertRaises(Exception, cassandraToNumpyArray, )
    #     self.assertRaises(Exception, myLeaveOneOut, )

    def test_numpyArrayToCassandra(self):
        # Check if all goes well in the normal expected case
        # With the 'with' statement used, this is a bit hairy
        mockcolumnfamily = Mock(spec=pycassa.ColumnFamily)
        mockcolumnfamily.batch = Mock()
        mockbatchinsert = Mock()
        mockbatchinsert.insert = mock_batchinsert
        mockbatch = Mock()
        mockbatch.__enter__ = Mock(return_value=mockbatchinsert)
        mockbatch.__exit__ = Mock(return_value=False)
        mockcolumnfamily.batch.return_value = mockbatch
        numpyArrayToCassandra(self.numpyArrayCorrect, mockcolumnfamily, self.keyCorrect)
        assert os.path.isfile('./'+self.keyCorrect+'-values.txt')
        assert os.path.isfile('./'+self.keyCorrect+'-shape.txt')


    def test_cassandraToNumpyArray(self):
        # Check if all goes well in the normal expected case
        # Create the files, first
        self.test_numpyArrayToCassandra()
        mockcolumnfamily = Mock(spec=pycassa.ColumnFamily)
        mockcolumnfamily.get = mock_columnfamilyget
        cassandraToNumpyArray(mockcolumnfamily, self.keyCorrect)

    def test_combinedCassandraNumpyArray(self):
        # Do both and check equality
        self.test_numpyArrayToCassandra()
        mockcolumnfamily = Mock(spec=pycassa.ColumnFamily)
        mockcolumnfamily.get = mock_columnfamilyget
        return_value = cassandraToNumpyArray(mockcolumnfamily, self.keyCorrect)
        assert return_value.all() == self.numpyArrayCorrect.all()


    def test_myLeaveOneOut(self):
        # This is not (yet?) testing for strange cases (where the rank of the matrix would be problematic, e.g.)
        # Set some data for testing (from stocks data set)
        X = np.loadtxt('./test_myLeaveOneOut_x.txt').reshape((633, 9))
        Y = np.loadtxt('./test_myLeaveOneOut_y.txt').reshape((633, 1))
        # Expected values computed using the original Matlab function
        expectedW = np.loadtxt('./test_myLeaveOneOut_W.txt')
        expectedErrLOO = np.loadtxt('./test_myLeaveOneOut_ErrLOO.txt')
        expectedYLOO = np.loadtxt('./test_myLeaveOneOut_YLOO.txt')

        W, YLOO, ErrLOO = myLeaveOneOut(X, Y)
        self.assertAlmostEqual(W.all(), expectedW.all())
        self.assertAlmostEqual(YLOO.all(), expectedYLOO.all())
        self.assertAlmostEqual(ErrLOO.all(), expectedErrLOO.all())



class ELMTest(unittest.TestCase):
    # We will test both FP and FN versions of the ELM
    def setUp(self):
        self.mockConnectionPool = Mock(spec=pycassa.pool.ConnectionPool)
        self.minimizationType = 'FP'
        self.numberNeurons = 100


    def test_init(self):
        # Check that wrong input raises exception
        self.assertRaises(Exception, ELM, 'SomeString', self.mockConnectionPool, self.numberNeurons)
        self.assertRaises(Exception, ELM, self.minimizationType, 'NotAConnectionPool', self.numberNeurons)
        self.assertRaises(Exception, ELM, self.minimizationType, self.mockConnectionPool, 'NotANumber')
        
    @patch('elm.pycassa.columnfamily.ColumnFamily')    
    def test_createTrainingData(self, mockFunction):
        returnedMock = Mock(spec=pycassa.ColumnFamily)
        returnedMock.get = mock_columnfamilyget
        returnedMock.get_range = mock_columnfamilyget_range
        returnedMock.batch = Mock()
        mockbatchinsert = Mock()
        mockbatchinsert.insert = mock_batchinsert
        mockbatch = Mock()
        mockbatch.__enter__ = Mock(return_value=mockbatchinsert)
        mockbatch.__exit__ = Mock(return_value=False)
        mockFunction.return_value = returnedMock
        returnedMock.batch.return_value = mockbatch
        self.myELM = ELM(self.minimizationType, self.mockConnectionPool, self.numberNeurons)
        self.myELM.createTrainingData()

    @patch('elm.pycassa.columnfamily.ColumnFamily')
    def test_storeTrainingDataToCassandra(self, mockFunction):
        returnedMock = Mock(spec=pycassa.ColumnFamily)
        returnedMock.get = mock_columnfamilyget
        returnedMock.get_range = mock_columnfamilyget_range
        returnedMock.batch = Mock()
        mockbatchinsert = Mock()
        mockbatchinsert.insert = mock_batchinsert
        mockbatch = Mock()
        mockbatch.__enter__ = Mock(return_value=mockbatchinsert)
        mockbatch.__exit__ = Mock(return_value=False)
        mockFunction.return_value = returnedMock
        returnedMock.batch.return_value = mockbatch
        self.myELM = ELM(self.minimizationType, self.mockConnectionPool, self.numberNeurons)
        self.myELM.createTrainingData()
        self.myELM.storeTrainingDataToCassandra()

    @patch('elm.pycassa.columnfamily.ColumnFamily')
    def test_train(self, mockFunction):
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
        self.myELM.train()


    @patch('elm.pycassa.columnfamily.ColumnFamily')
    def test_verdict(self, mockFunction):
        returnedMock = Mock(spec=pycassa.ColumnFamily)
        returnedMock.get = mock_columnfamilyget
        mockFunction.return_value = returnedMock

        self.myELM = ELM(self.minimizationType, self.mockConnectionPool, self.numberNeurons)
        self.myELM.train()

        inputTestData = np.loadtxt('./test_input_data_training_elm.txt').reshape((633, 9))
        outputTestData = np.loadtxt('./test_output_data_training_elm.txt').reshape((633, 1))
        ELMTestingData = ELMData(inputTestData, outputTestData, [], [])



if __name__ == '__main__':
    unittest.main()