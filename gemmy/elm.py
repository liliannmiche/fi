import numpy as np
from copy import copy

import pycassa

import cPickle as Pickle



class ELMData(object):
    """ This class mainly checks for data structure so that the ELM class
        can run without too much data checking.
    """

    def __init__(self, inputs, outputs, means, stds):
        if not isinstance(inputs, np.ndarray):
            raise Exception('Given input data is not a Numpy array.')
        if not isinstance(outputs, np.ndarray):
            raise Exception('Given output data is not a Numpy array.')
        if not len(inputs.shape) == 2:
            raise Exception('Given input data is not 2-dimensional.')
        if not len(outputs.shape) == 2:
            raise Exception('Given output data is not 2-dimensional.')
        if not outputs.shape[1] == 1:
            raise Exception('Given output is multi-dimensional, not handled.')
        if not inputs.shape[0] == outputs.shape[0]:
            raise Exception('Given input and output do not have the same number of samples')
        if not len(means)==0:
            if not len(means)==inputs.shape[1]:
                raise Exception('Given means vector is not of the right size (expected %s, got %s).' % (inputs.shape[1], len(means)))
        if not len(stds)==0:
            if not len(stds)==inputs.shape[1]:
                raise Exception('Given stds vector is not of the right size (expected %s, got %s).' % (inputs.shape[1], len(stds)))
        self.inputNumSamples, self.inputDimensionality = inputs.shape
        self.outputNumSamples, self.outputDimensionality = outputs.shape
        self.input = inputs
        self.output = outputs
        self.means = means
        self.standardDeviations = stds
        self.normalized = False
        if (len(means)==0 & len(stds)==0):
            for i in range(self.inputDimensionality):
                self.means.append(np.mean(self.input[:, i]))
                self.standardDeviations.append(np.std(self.input[:, i]))

    def normalizeData(self, means=None, standardDeviations=None):
        # Means and Standard Deviations are provided for normalization
        if means is None:
            means=self.means
        if standardDeviations is None:
            standardDeviations = self.standardDeviations
        if 0.0 in standardDeviations:
            indicesZeroStd = standardDeviations.index(0.0)
            raise Exception('Given standard deviation is zero for variables %s' % indicesZeroStd)
        if not len(standardDeviations) == self.input.shape[1]:
            raise Exception('Given standard deviations miss values (expected %s, got %s).' % (self.input.shape[1], len(standardDeviations)))
        if not len(means) == self.input.shape[1]:
            raise Exception('Given means miss values (expected %s, got %s).' % (self.input.shape[1], len(means)))
        for i in range(self.inputDimensionality):
            self.input[:, i] = (self.input[:, i]-means[i])/standardDeviations[i]  
        self.normalized = True



def numpyArrayToCassandra(numpyArray, columnFamily, key):
    """ Puts a Numpy array in a Cassandra key. Uses Pickle to 
        store the matrix to a string and back.
    """
    if not isinstance(columnFamily, pycassa.ColumnFamily):
        raise Exception('Given Column Family is not an instance of a Column Family.')
    if not isinstance(numpyArray, np.ndarray):
        raise Exception('Given array is not a Numpy Array.')
    if not isinstance(key, str):
        raise Exception('Given key is not a string.')

    pickledValues = Pickle.dumps(numpyArray.ravel())
    pickledShape = Pickle.dumps(numpyArray.shape)

    with columnFamily.batch(queue_size=2) as columnFamilyBatch:
        columnFamilyBatch.insert(key, {'values': pickledValues})
        columnFamilyBatch.insert(key, {'shape': pickledShape})


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

def myLeaveOneOut(X, Y):
    # Verify inputs
    if not isinstance(X, np.ndarray):
        raise Exception('Given array is not a Numpy Array.')
    if not isinstance(Y, np.ndarray):
        raise Exception('Given array is not a Numpy Array.')
    if not len(X.shape) == 2:
        raise Exception('Given X input data is not a 2D matrix.')
    if not len(Y.shape) == 2:
        raise Exception('Given Y output data is not a 2D matrix.')
    if not X.shape[0] == Y.shape[0]:
        raise Exception('Number of samples of input X and output Y do not match.')
    if not Y.shape[1] == 1:
        raise Exception('Multi-dimensional output not supported in this function.')

    # Directly translated from the matlab function of the same name.
    # Optimizations likely possible :)
    N, d = X.shape
    C = np.dot(X.T, X)
    XCondNumber = 1/np.linalg.cond(X)

    # If the input matrix is normally conditioned
    if XCondNumber > 10^(-14):
        W, residues, rank, s = np.linalg.lstsq(X, Y)
        P = np.dot(X, np.linalg.inv(C))
        proxyDiagMatrix = np.diag(np.dot(P, X.T))
        S = Y-np.dot(X, W)
        S = S/(1-proxyDiagMatrix)
        ErrLOO = np.mean(np.power(S, 2))
        YLOO = Y - S;
    # Otherwise, we just don't perform the LOO
    else:
        W = np.zeros((d, 1))
        YLOO = np.zeros((N, 1))
        ErrLOO = np.inf

    return W, YLOO, ErrLOO


def findallequal(mylist, value=1):
    # Returns the list of indices of the elements equal to value
    return [i for i in xrange(len(mylist)) if mylist[i]==value]


def argsortedlist(mylist, sorting='ascend'):
    """ Returns a list of indices of the sorted values in the list
    """
    newlist = zip(mylist, xrange(len(mylist)))
    if sorting == 'ascend':
        newlist.sort()
    else:
        newlist.sort(reverse=True)

    arguments = [item[1] for item in newlist]
    return arguments

   

class ELM(object):
    """
        Class implementing the Extreme Learning Machine with False Positives/Negatives
        optimization and Leave-One-Out based truncated Forward Feature Selection (yes, this is long).

        Parameters
        ----------
        minimizationType : str in ['FP', 'FN']
        connectionPool : pycassa.pool.ConnectionPool object
        NumberNeurons : int, optional (strictly positive)

        Attributes
        ----------
        trained            : bool
                             Tells whether the ELM has been trained (i.e. it can be used to produce a verdict directly).
        pool               : pycassa.pool.ConnectionPool
                             The connection pool to a Cassandra cluster used for fetching/writing ELM-related data.
        numberNeurons      : int
                             Given number of neurons
        minimizationType   : str in ['FP', 'FN']
                             Type of minimization thr ELM will perform
        activationFunction : numpy.tanh
                             Activation function for the neurons. Set to this arbitrarily.
        cfELM              : pycass.columnfamily.ColumnFamily
                             The column family in which the ELM data is written/fetched from.
        inputWeights       : numpy.array (double)
                             Contains the input weights of the trained ELM (affected only after training).
        inputBiases        : numpy.array (double)
                             Contains the input biases of the trained ELM (affected only after training).
        outputWeights      : numpy.array (double)
                             Contains the output weights of the trained ELM (affected only after training). 
        bestNeurons        : numpy.array (double)
                             The list of the best neurons selected for the task (affected only after training).


        Initializing an ELM object only affects some self values, *IT DOES NOT TRAIN IT*.
        
    """
    def __init__(self, minimizationType, connectionPool, NumberNeurons=100):
        # Assumes we have an initial ELM in a Cassandra CF
        # Tries to get the weights from the column Family
        if not isinstance(connectionPool, pycassa.pool.ConnectionPool):
            raise Exception('Given pool is not a proper pycassa Connection Pool.')
        if not minimizationType in ['FP', 'FN']:
            raise Exception('Given minimizationType is not supported (should be FP or FN), got %s' % minimizationType)
        if not isinstance(NumberNeurons, int):
            raise Exception('Given number of neurons is not an integer.')
        if not NumberNeurons > 0:
            raise Exception('Given number of neurons is not (strictly) positive.')
        self.pool = connectionPool
        self.numberNeurons = NumberNeurons
        self.minimizationType = minimizationType
        self.activationFunction = np.tanh
        self.trained = False
        self.cfELM = pycassa.columnfamily.ColumnFamily(self.pool, 'elm'+self.minimizationType)


    def train(self):
        """
            The main function to be called for training/fetching a trained ELM:
                - Checks if ELM structure exists in Cassandra
                - If it exists, load it to self and declare ELM as trained
                - If not: 
                    - Proceed to creating the training data (triplets and class)
                    - Store that training data to Cassandra in directly usable form
                    - Perform the actual training on the training dat just obtained
                    - Store the trained ELM in Cassandra
        """
        try:
            # If something is present in the associated Column Family, load that data
            self.fetchELMComponentsFromCassandra()
            # TODO: Do some checking over the pulled data here (size checks)
            self.trained = True
        except pycassa.NotFoundException:
            # If we cannot find some of the required data for the ELM, we train it
            # and store to Cassandra
            self.createTrainingData()
            self.storeTrainingDataToCassandra()
            self.performTraining()
            self.trained = True


    def fetchTrainingDataFromCassandra(self):
        """ 
            Fetches the latest training input and output from Cassandra
        """
        self.ELMTrainingData = ELMData(cassandraToNumpyArray(self.cfELM, 'trainingInput'), cassandraToNumpyArray(self.cfELM, 'trainingOutput'), list(cassandraToNumpyArray(self.cfELM, 'trainingMeans')), list(cassandraToNumpyArray(self.cfELM, 'trainingStds')))

    def createTrainingData(self):
        """ 
            Builds the input X and output Y data of the form:
            X = [dNN_i, dNNOpp_i, RNNOpp_i] and Y = [Class_i]
            from the data held in the Cassandra column family rawtrainingdata
        """
        cfRawTrainingData = pycassa.columnfamily.ColumnFamily(self.pool, 'rawtrainingdata')
        rawTrainingDataIterator = cfRawTrainingData.get_range(columns=['NN_dist', 'NNd_dist', 'NNd_rank', 'CHANGEME'], read_consistency_level=pycassa.ConsistencyLevel.ONE)
        inputDataList = []
        outputDataList = []
        for item in rawTrainingDataIterator:
            inputDataList.append([float(item[1].values()[0]), float(item[1].values()[1]), float(item[1].values()[2]]))
            outputDataList.append(float(item[1].values()[3]))
        inputData = np.array(inputDataList).reshape((len(inputDataList), 3))
        outputData = np.array(outputDataList).reshape((len(outputDataList), 1))
        self.ELMTrainingData = ELMData(inputData, outputData, [], [])

    def insert_triple(self, valuesDict):
        if not isinstance(valuesDict, dict):
            raise Exception('Given data is not in a dictionary form.')
        if not 'NN_dist' in valuesDict.keys():
            raise Exception('')
        if not 'NNd_dist' in valuesDict.keys();:
            raise Exception('')
        if not 'NNd_rank' in valuesDict.keys():
            raise Exception('')
        if not 'CHANGEME' in valuesDict.keys():
            raise Exception('')
        cfRawTrainingData = pycassa.columnfamily.ColumnFamily(self.pool, 'rawtrainingdata')
        cfRawTrainingData.insert(sampleSHA1, valuesDict)


    def storeTrainingDataToCassandra(self):
        """ 
            Writes the training input and output (as well as means and stds) to Cassandra for keeping
        """
        numpyArrayToCassandra(self.ELMTrainingData.input, self.cfELM, 'trainingInput')
        numpyArrayToCassandra(self.ELMTrainingData.output, self.cfELM, 'trainingOutput')
        numpyArrayToCassandra(np.array(self.ELMTrainingData.means), self.cfELM, 'trainingMeans')
        numpyArrayToCassandra(np.array(self.ELMTrainingData.standardDeviations), self.cfELM, 'trainingStds')


    def storeELMComponentsToCassandra(self):
        """ 
            Writes the trained ELM weights and biases to Cassandra:
            self.inputWeights, self.inputBiases, self.outputWeights
        """
        numpyArrayToCassandra(self.inputWeights, self.cfELM, 'inputWeights')
        numpyArrayToCassandra(self.outputWeights, self.cfELM, 'outputWeights')
        numpyArrayToCassandra(self.inputBiases, self.cfELM, 'inputBiases')
        numpyArrayToCassandra(self.bestNeurons, self.cfELM, 'bestNeurons')


    def fetchELMComponentsFromCassandra(self):
        """ 
            Fetches the ELM components from Cassandra and affects self.
            The requests to Cassandra will raise an exception if the data is not available in Cassandra.
            That exception gets intercepted in the 'train' function.
        """
        self.inputWeights = cassandraToNumpyArray(self.cfELM, 'inputWeights')
        self.inputBiases = cassandraToNumpyArray(self.cfELM, 'inputBiases')
        self.outputWeights = cassandraToNumpyArray(self.cfELM, 'outputWeights')
        self.bestNeurons = cassandraToNumpyArray(self.cfELM, 'bestNeurons')
        self.activationFunction = np.tanh
        self.numberNeurons = self.inputWeights.shape[0]
        self.fetchTrainingDataFromCassandra()


    def performTraining(self):
        """
            This function trains the ELM object, i.e.:
                - Assumes there is training data available to train it (triplets [dNN_i, dNNOpp_i, RNNOpp_i] and [Class_i])
                - Fetches the data from the Cassandra node (from CF named 'elmFP' or 'elmFN')
                - Normalizes the training data (zero mean, unit variance)
                - Runs the training part (generates random matrices, finds the best neurons combination, etc.)
                - Stores the trained ELM (matrices) to Cassandra by serializing matrices to strings (using Pickle)
        """
        # First fetch the training data from Cassandra nodes
        self.fetchTrainingDataFromCassandra()

        # Normalize the data (zero mean, unit variance)
        self.ELMTrainingData.normalizeData()

        # Actual training using available training data
        # Generate random input weights
        self.inputWeights = np.random.randn(self.numberNeurons, self.ELMTrainingData.inputDimensionality)*np.sqrt(3)
        # Generate random input biases
        self.inputBiases = np.random.randn(self.numberNeurons, 1)*0.5
        # Project the input data
        self.hiddenLayer = np.dot(self.ELMTrainingData.input, self.inputWeights.T)
        # Add the biases
        self.hiddenLayer += np.tile(self.inputBiases, (1, self.hiddenLayer.shape[0])).T
        # Non-linearity in the neurons (tanh function)
        self.hiddenLayer = self.activationFunction(self.hiddenLayer)
        # Add linear neurons to the total
        self.hiddenLayer = np.hstack((self.hiddenLayer, self.ELMTrainingData.input))

        # Up to here, this is the normal ELM
        # Now we evaluate a weighted classification Leave One Out error for various neuron configurations
        # in order to retain the best neurons for the specific minimization task
        totalNumberNeurons = self.hiddenLayer.shape[1]
        bestClassificationRateLOO = 0
        indices = [0]*totalNumberNeurons
        indicesCleanSamples, dump = np.nonzero(self.ELMTrainingData.output == -1) 
        indicesMalwareSamples, dump = np.nonzero(self.ELMTrainingData.output == 1)
        bestNeurons = []


        # Check the minimization type and affect error criterion factors accordingly
        # The factors are arbitrary here. The ratio between the two weights just has to be "large enough"
        # Experimentally, this seemed a good value
        if self.minimizationType == 'FP':
            fpFactor = 9.0
            fnFactor = 1.0
        else:
            fnFactor = 9.0
            fpFactor = 1.0


        # Loop at least that many times to find a good combination of neurons
        # This is actually a "truncated forward selection" algorithm:
        #   - Find the best neuron alone that gives the max Classification rate in LOO (weighted)
        #   - Keep that best neuron and find the one in the remaining ones that combines best with the first found
        #   - Keep these best two neurons and find the one in the remaining ones that combines best with the two found
        #   - ... until you reach 20 neurons (arbitrary, can be larger if time allows)
        for j in xrange(min(20, totalNumberNeurons)):
            # Loop a maximum of 20 times (to find maximum 20 neurons)
            ClassificationRatesLOO = [0.0]*totalNumberNeurons
            # Loop over all the existing neurons
            for i in xrange(totalNumberNeurons):
                # If we have selected that neuron before, pretend the classification rate is 0 to never select that neuron
                if indices[i] == 1:
                    ClassificationRatesLOO[i] = 0.0
                else:
                    # Otherwise, evaluate the Classification error in LOO of Type 1 (on clean samples) and of Type 2 (on malware samples)
                    tempIndices = copy(indices)
                    tempIndices[i] = 1
                    betas, yloo, dump = myLeaveOneOut(self.hiddenLayer[:, findallequal(tempIndices)], self.ELMTrainingData.output)

                    ClassificationRateLOOType1 = np.mean(self.ELMTrainingData.output[indicesCleanSamples, 0]==np.sign(yloo[indicesCleanSamples, 0]))

                    ClassificationRateLOOType2 = np.mean(self.ELMTrainingData.output[indicesMalwareSamples, 0]==np.sign(yloo[indicesMalwareSamples, 0]))
                    # The final optimized classification rate in LOO is the weighted version of the Type 1 and Type 2
                    ClassificationRatesLOO[i] = (fpFactor*ClassificationRateLOOType1+fnFactor*ClassificationRateLOOType2)/(fpFactor+fnFactor)

            # Find the neuron that works best with the already selected ones (max of weighted classification rate)
            maxClassificationRateLOO = np.max(ClassificationRatesLOO)
            argsortedClassificationRateLOO = argsortedlist(ClassificationRatesLOO, sorting='descend')
            indexMaxClassificationRateLOO = argsortedClassificationRateLOO[0]
            # Retain that best neuron in the list of the retained neurons
            indices[indexMaxClassificationRateLOO] = 1

            # Write the best neurons indices to the object and compute the output weights of the model for that specific set of best neurons
            if bestClassificationRateLOO<maxClassificationRateLOO:
                bestClassificationRateLOO = maxClassificationRateLOO
                self.bestClassificationRateLOO = bestClassificationRateLOO
                self.bestNeurons = np.array(indices)
                betasfinal, yloofinal, dump2 = myLeaveOneOut(self.hiddenLayer[:, findallequal(self.bestNeurons)], self.ELMTrainingData.output)


        # Set the output weights 
        self.outputWeights = betasfinal

        self.trained = True

        # Write the trained ELM components to Cassandra
        self.storeELMComponentsToCassandra()


    def verdict(self, testData):
        """ 
            Returns a verdict (-1 or 1) on the given test data, along with some sort of confidence.

            Parameters
            ----------
            testData : ELMData object
                       The ELMData object containing the data to test (i.e. with a bogus output value!)

            Returns
            -------
            verdict    : int
                         Returns -1 for clean and 1 for malware
            confidence : double
                         Returns a number between 0 and 1. Numbers close to 1 are supposed to be higher confidence decisions.

            Requires the ELM object to have been trained.
        """
        # Give a verdict on the given test data
        if not isinstance(testData, ELMData):
            raise Exception('Given test data is not an ELMData object.')
        if not self.trained:
            raise Exception('ELM has not been trained.')
        # Test here
        # Start by normalizing the input test data with the training factors
        testData.normalizeData(self.ELMTrainingData.means, self.ELMTrainingData.standardDeviations)
        hiddenLayerTest = np.dot(testData.input, self.inputWeights.T)
        hiddenLayerTest += np.tile(self.inputBiases, (1, hiddenLayerTest.shape[0])).T
        hiddenLayerTest = self.activationFunction(hiddenLayerTest)
        hiddenLayerTest = np.hstack((hiddenLayerTest, testData.input))
        hiddenLayerTest = hiddenLayerTest[:, findallequal(self.bestNeurons)]
        
        confidence = np.dot(hiddenLayerTest, self.outputWeights)
        verdict = np.sign(confidence)
        confidence = np.tanh(np.abs(confidence))

        return confidence, verdict
        

