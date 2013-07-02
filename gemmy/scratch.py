"""
    A python implementation of the Extreme Learning Machine to work with
    Cassandra backend.
    Stores the weights and biases in a column family, and retrieves them
    for testing.
"""

import numpy as np

import pycassa

import cPickle as Pickle


from config import *




class ELMData(object):
    """ This class mainly checks for data structure so that the ELM class
        can run without too much data checking.
    """

    def __init__(self, inputs, outputs):
        if not isinstance(inputs, np.ndarray):
            raise Exception('Given input data is not a Numpy array.')
        if not isinstance(outputs, np.ndarray):
            raise Exception('Given output data is not a Numpy array.')
        if not len(inputs.shape) == 2:
            raise Exception('Given input data is not 2-dimensional.')
        if not len(outputs.shape) == 2:
            raise Exception('Given output data is not 2-dimensional.')
        self.inputNumSamples, self.inputDimensionality = inputs.shape
        self.outputNumSamples, self.outputDimensionality = outputs.shape
        self.input = inputs
        self.output = outputs
        self.means = []
        self.standardDeviations = []
        self.normalized = False
        for i in range(self.inputDimensionality):
            self.means.append(np.mean(self.input[:, i]))
            self.standardDeviations.append(np.std(self.input[:, i]))

    def _normalizeData(self):
        # Normalize the data
        for i in range(self.inputDimensionality):
            self.input[:, i] = (self.input[:, i]-self.means[i])/self.standardDeviations[i]  
        self.normalized = True


class ELMTrainData(ELMData):

    _dictOutputTypes = {'regression':   'regression',
                        'r':            'regression',
                        'reg':          'regression',
                        'classification':   'classification', 
                        'c':                'classification',
                        'classif':          'classification'
                        }

    def __init__(self, inputs, outputs, outputType='r'):
        super(ELMTrainData, self).__init__(inputs, outputs)
        if not outputType in self._dictOutputTypes.keys():
            raise Exception('Given problem type not supported (got '+str(outputType)+', expects one of '+str(self._dictOutputTypes.keys())+').')
        self.outputType = self._dictOutputTypes[outputType]
        if not self.inputNumSamples == self.outputNumSamples:
            raise Exception('Input number of samples does not match output.')
        self._normalizeData()


class ELMTestData(ELMData):

    def __init__(self, inputs, outputs=np.zeros((0, 0))):
        super(ELMTestData, self).__init__(inputs, outputs)
        if self.outputNumSamples != 0:
            self.hasOutput = True
        self._normalizeData()


def numpyArrayToCassandra(numpyArray, columnFamily, key):
    """ Puts a Numpy array in a Cassandra key. Uses Pickle to 
        store the matrix to a string and back.
    """
    if not isinstance(columnFamily, pycassa.ColumnFamily):
        raise Exception('Given Column Family is not an instance of a Column Family.')
    if not isinstance(numpyArray, numpy.array):
        raise Exception('Given array os not a Numpy Array.')
    if not isinstance(key, str):
        raise Exception('Given key is not a string.')
    with columnFamily.batch(queue_size=2) as columnFamilyBatch:
        columnFamilyBatch.insert(key, {'values': Pickle.dumps(numpyArray.ravel())})
        columnFamilyBatch.insert(key, {'shape': Pickle.dumps(numpyArray.shape)})


def cassandraToNumpyArray(columnFamily, key):
    """ Returns a previously stored Numpy array from Cassandra.
    """
    if not isinstance(columnFamily, pycassa.ColumnFamily):
        raise Exception('Given Column Family is not an instance of a Column Family.')
    if not isinstance(key, str):
        raise Exception('Given key is not a string.')
    data = columnFamily.get(key, columns=['shape', 'values'])
    numpyArray = Pickle.loads(data['values']).reshape(Pickle.loads(data['shape']))
    return numpyArray


class ELM(object):

    _dictActivationFunctions = {'np.sin': np.sin, 'np.tanh': np.tanh}

    def __init__(self, connectionPool):
        # Tries to get the weights from the column Family
        self.trained = False
        try:
            self.pool = connectionPool
            cfELM = pycassa.ColumnFamily(connectionPool, 'elm')
            self.inputWeights = cassandraToNumpyArray(cfELM, 'inputWeights')
            self.inputBiases = cassandraToNumpyArray(cfELM, 'inputBiases')
            self.outputWeights = cassandraToNumpyArray(cfELM, 'outputWeights')
            self.activationFunction = cfELM.get('activationFunction', columns=['activationFunction'])
            self.activationFunction = self._dictActivationFunction[self.activationFunction['activationFunction']]
            self.numberNeurons = self.inputWeights.shape[0]
            self.trained = True
        except NotFoundException:
            # If we cannot find some of the required data for the ELM, we train it
            # and store to Cassandra
            print 'No ELM data found in database, proceeding to training.'
            self.train()
            self.trained = True

    def getNearestNeighborOppositeClass(self, neighbors, classif):
        if classif == 1:
            # We are looking for the nearest clean sample
            oppClassif = -1
        else:
            # We look for the nearest malware
            oppClassif = 1

        rank, distance = [(i,neighbors[i][0]) for i, item2 in enumerate([item[1] for item in neighbors if item[1]==oppClassif])][0]
        return rank, distance

    def getNearestNeighbor(self, neighbors):
        classif = neighbors[0][1]
        distance = neighbors[0][0]
        return classif, distance

    def createTrainingData(self):
        # Create a special Gemmy instance for computing the nearest neighbors
        # for the ELM training
        gemmyInstance = gemmy.Gemmy(ELMTraining=True)
        trainData = np.zeros((len(listSamples, 3)))
        for sample in listSamples:
            task = gemmy.GemmyTask(sample)
            gemmyInstance.run(task)
            neighbors = gemmyInstance.neighbors
            cfFeaturestats = pycassa.ColumnFamily(self.pool, 'featurestats')
            classes = cfFeaturestats.multiget([item[0] for item in neighbors], columns=['classification'])
        self.trainData = trainData

    def getNeighbors(self):

        return neighbors


    def elmTrain(self, numSamples=confELMNumSamples, \
                       maxRefSamplesCheckedAgainst=confMaxRefSamplesCheckedAgainst, \
                       numberNeurons=confELMNumberNeurons, \
                       activationFunction=confELMActivationFunction):
        if not isinstance(numSamples, int):
            raise Exception('Given number of samples for training is not an integer.')

        self.createTrainingData()

        # Initialize empty training data
        elmTrainData.input = np.zeros((numSamples, 3))
        elmTrainData.inputDimensionality = 3
        elmTrainData.output = np.zeros((numSamples, 1))

        #
        # Train here:
        self.inputWeights = np.random.random((self.numberNeurons, self.elmTrainData.inputDimensionality))*2-1
        self.inputBiases = np.random.random((self.numberNeurons, 1))
        self.hiddenLayer = np.dot(self.elmTrainData.input, self.inputWeights.T)
        self.hiddenLayer += np.tile(self.inputBiases, (1, self.hiddenLayer.shape[0])).T
        self.hiddenLayer = self.activationFunction(self.hiddenLayer)
        self.outputWeights = np.dot(np.linalg.pinv(self.hiddenLayer), self.elmTrainData.output)
        self.trainOutput = np.dot(self.hiddenLayer, self.outputWeights)
        if self.elmTrainData.outputType == 'regression':
            self.trainError = np.mean(np.power(self.trainOutput-self.elmTrainData.output, 2), axis=0)
        self.trained = True

    def print_errors(self):
        print 'Training Error: '+str(self.trainError)
        print 'Testing Error: '+str(self.testMSE)

    def test(self, elmTestData):
        if not self.trained:
            raise Exception('ELM has not been trained.')
        if not isinstance(elmTestData, ELMTestData):
            raise Exception('Given data for testing is not an ELMTestData object.')
        self.elmTestData = elmTestData
        # Test here
        self.hiddenLayerTest = np.dot(self.elmTestData.input, self.inputWeights.T)
        self.hiddenLayerTest += np.tile(self.inputBiases, (1, self.hiddenLayerTest.shape[0])).T
        self.hiddenLayerTest = self.activationFunction(self.hiddenLayerTest)
        self.testOutput = np.dot(self.hiddenLayerTest, self.outputWeights)
        # If we have an output, let's compute the MSE/classification error
        if self.elmTestData.hasOutput:
            if self.elmTrainData.outputType == 'regression':
                self.testMSE = np.mean(np.power(self.testOutput-self.elmTestData.output, 2), axis=0)
            else:
                raise NotImplementedError

