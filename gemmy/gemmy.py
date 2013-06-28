"""
    The gemmy module whould be used this way::
        
        >>> import gemmy
        >>> gemmyInstance = gemmy.Gemmy()
        >>> task = gemmy.GemmyTask(JSONData)
        >>> SMAFact = gemmyInstance.run(task)

    With *JSONData* being a dictionary provided by the pre-processor with a specific structure:

    .. code-block:: javascript
                
        { 'priority': int, 
          'sha1': str, 
          'preprocessor_data': { 'classification_status': str, 
                                 'pp_results': { '59': ['0346383fa3e', '0346383fa3e'],
                                                 '28': ['237896fe2a', '237896fe2a'],
                                                 ...,
                                               }
                               } , 
          'metadata': { 'sha1': str,
                        'system': str,
                        'first_seen': str,
                        'sha256': str,
                        'md5': str,
                        'size': int,
                        'classification': str
                        
                      }
        }

"""


# Global imports
import bisect
import json
import time
import numpy as np

import cPickle as Pickle

# Cassandra API
import pycassa


# Default value parameters for algorithm and Cassandra connection
from config import *


class GemmyTask(object):
    """
        This class is used to check the given data held in a dictionary.

        The data is put into a dictionary format expected by the other functions.

        Most methods are private.

        :rtype: A :class:`GemmyTask` object, to be used by a \
        :class:`Gemmy` instance (essentially holds the pre-processor and metadata in a dictionary format)

        .. document private functions
        .. automethod:: _checkPreprocessorData
        .. automethod:: _checkMetadata
        .. automethod:: _checkTaskDictionary
       
    """

    # Lists of required fields for the given task data
    _listClassificationStatuses = ['', 'unknown', 'reference']
    _requiredPPKeys = ['classification_status', 'pp_results']
    _requiredTaskKeys = ['priority', 'sha1', 'preprocessor_data', 'metadata']
    _requiredMetadataKeys = ['sha1', 'system', 'first_seen', 'sha256', 'md5', 'size', 'classification']

    def _checkPreprocessorData(self, PPDictionary):
        """
            Checks the pre-processor dictionary data.
            Has to provide the _requiredPPKeys keys and have a known classification status.
            Affects the object directly.
        """
        if not isinstance(PPDictionary, dict):
            raise Exception('Given preprocessor data is not in a dictionary format.')
        for key in self._requiredPPKeys:
            if key not in PPDictionary.keys():
                raise Exception('Given preprocessor data misses required keys (requires: '+str(self._requiredPPKeys)+', missing '+str(key)+').')
        classification_status = PPDictionary['classification_status']
        if not isinstance(classification_status, str):
            raise Exception('Given classification status is not a string.')
        if PPDictionary['classification_status'] not in self._listClassificationStatuses:
            raise Exception('Given classification status is not supported (supports: '+str(self._listClassificationStatuses)+').')
        self.classification_status = PPDictionary['classification_status']
        # Do some checking on the given PP data?
        for key in PPDictionary['pp_results'].keys():
            featureNumber = key
            featureValuesList = PPDictionary['pp_results'][key]
            for featureValue in featureValuesList:
                if ',' in featureValue:
                    featureValue = featureValue.split(',')[1]
                if not self.pp_results.has_key(str(featureNumber)):
                    self.pp_results[str(featureNumber)] = []
                self.pp_results[str(featureNumber)].append(featureValue)

    def _checkMetadata(self, MDictionary):
        """
            Checks the metadata format and values.
            Has to provide the _requiredMetadataKeys keys.
            Affects the object directly.
        """
        if not isinstance(MDictionary, dict):
            raise Exception('Given metadata is not in a dictionary format.')
        for key in self._requiredMetadataKeys:
            if key not in MDictionary.keys():
                raise Exception('Given metadata misses required keys (requires: '+str(self._requiredMetadataKeys)+', missing '+str(key)+').')
        if not isinstance(MDictionary['system'], str):
            raise Exception('Given system value is not a string.')
        self.system = MDictionary['system']
        if not isinstance(MDictionary['classification'], str):
            raise Exception('Given classification value is not a string.')
        self.classification = MDictionary['classification']
        if not isinstance(MDictionary['first_seen'], str):
            raise Exception('Given first seen is not a string.')
        self.first_seen = MDictionary['first_seen']
        if not isinstance(MDictionary['sha256'], str):
            raise Exception('Given sha256 is not a string.')
        self.sha256 = MDictionary['sha256']
        self.sha1 = MDictionary['sha256']
        if not isinstance(MDictionary['md5'], str):
            raise Exception('Given md5 is not a string.')
        self.md5 = MDictionary['md5']
        if not isinstance(MDictionary['size'], int):
            raise Exception('Given file size is not an integer.')
        self.size = MDictionary['size']

    def _checkTaskDictionary(self, taskDictionary):
        """
            Checks the global given dictionary.
            Calls the subfunctions checking the pre-processor data and metadata.
            Affects the object directly.
        """
        for key in self._requiredTaskKeys:
            if key not in taskDictionary.keys():
                raise Exception('Given task misses required keys (requires: '+str(self._requiredTaskKeys)+', missing '+str(key)+').')
        priority = taskDictionary['priority']
        if not isinstance(priority, int):
            raise Exception('Given task priority is not an integer.')
        sha1 = taskDictionary['sha1']
        if not isinstance(sha1, str):
            raise Exception('Given SHA1 is not a string.')
        self._checkPreprocessorData(taskDictionary['preprocessor_data'])
        self._checkMetadata(taskDictionary['metadata'])

    def __init__(self, taskDictionary):
        if not isinstance(taskDictionary, dict):
            raise Exception('Given task is not in a dictionary format.')
        self.priority = 0
        self.sha1 = ""
        self.classification_status = ""
        self.pp_results = {}
        self.system = ""
        self.first_seen = ""
        self.sha256 = ""
        self.md5 = ""
        self.size = 0
        self._checkTaskDictionary(taskDictionary)



    def isReference(self):
        """
            :rtype: boolean

            Returns whether the task is a reference sample (i.e. meant to be \
            inserted into the reference set) or an unknown sample (on which to take a decision).
        """
        if self.classification_status in classificationNamesReference:
            return True
        else:
            return False
    



class Gemmy(object):
    """ 
        Initializes a Gemmy object with Cassandra and internal algorithm parameters. Default parameter values are configured \
        in file *config.py*. All parameters are optional and will default to the configured values from *config.py*:

        :param keySpace: Keyspace used to store the information about the reference set (MinHash data and sample statistics)
        :type keySpace: str [defaults to *confCassandraKeySpace*]
        :param serverList: List of Cassandra hosts to connect to. Must be in the format ['host1:port1', 'host2:port2']
        :type serverList: list [defaults to *confCassandraHostAndPort*]
        :param poolSize: Size of the pool used for the Cassandra requests
        :type poolSize: int [defaults to *confCassandraPoolSize*]
        :param timeOut: Maximum time to wait before declaring a pycassa.TimeOut exception
        :type timeOut: float or ``None`` [defaults to *confCassandraTimeOut*]
        :param maxNumberHashes: Number of hashes to use for the MinHash approximation
        :type maxNumberHashes: int [defaults to *confRefSamplesMaxNumberHashes*]
        :param usedFeatures: Feature numbers to use (passed as a list of strings)
        :type usedFeatures: list [defaults to *confUsedFeatures*]
        :param maxNumberNeighbors: Number of nearest neighbors to return
        :type maxNumberNeighbors: int [defaults to *confNumberNeighborsReturned*]
        :param maxRefSamplesCheckedAgainst:  Number of reference samples against which we check
        :type maxRefSamplesCheckedAgainst: int [defaults to *confMaxRefSamplesCheckedAgainst*]
        :rtype: Gemmy object to be used on a :class:`gemmy.GemmyTask` object using :func:`run`

        .. document private functions
        .. automethod:: _insertReferenceSampleInCassandra
        .. automethod:: _getDecision
        .. automethod:: _takeEnsembleDecision
        .. automethod:: _takeDecisionClassifier1
        .. automethod:: _takeDecisionClassifier2

    """

    
    def __init__(self,  keySpace=confCassandraKeySpace,\
                        serverList=confCassandraHostAndPort,\
                        poolSize=confCassandraPoolSize,\
                        timeOut=confCassandraTimeOut,\
                        maxNumberHashes=confRefSamplesMaxNumberHashes,\
                        usedFeatures=confUsedFeatures,\
                        maxNumberNeighbors=confNumberNeighborsReturned,\
                        maxRefSamplesCheckedAgainst=confMaxRefSamplesCheckedAgainst):
        self.keySpace = keySpace
        self.serverList = serverList
        self.poolSize = poolSize
        self.timeOut = timeOut
        self.pool = pycassa.pool.ConnectionPool(self.keySpace,\
                                            server_list=self.serverList,\
                                            pool_size=self.poolSize,
                                            timeout=self.timeOut)
        self.hashFunction = lambda x:x
        self.maxNumberHashes = maxNumberHashes
        self.usedFeatures = usedFeatures
        self.maxNumberNeighbors = maxNumberNeighbors
        self.maxRefSamplesCheckedAgainst = maxRefSamplesCheckedAgainst
        self.refFilesHashes = []
        if __debug__: print "Generating Reference Set file list..."
        cfFeaturestats = pycassa.ColumnFamily(self.pool, 'featurestats')
        fileHashesIterator = cfFeaturestats.get_range(row_count=self.maxRefSamplesCheckedAgainst,columns=['system'],read_consistency_level=pycassa.ConsistencyLevel.ONE)
        for item in fileHashesIterator:
            self.refFilesHashes.append(item[0])
        
        # ELM object section
        # Tries to get the weights from the column Family
        self.elmTrained = False
        try:
            if __debug__: print 'Trying for ELM models in database...'
            cfELM = pycassa.ColumnFamily(self.pool, 'elmFP')
            self.elmFPInputWeights = self._cassandraToNumpyArray(cfELM, 'inputWeights')
            self.elmFPInputBiases = self._cassandraToNumpyArray(cfELM, 'inputBiases')
            self.elmFPOutputWeights = self._cassandraToNumpyArray(cfELM, 'outputWeights')
            self.elmFPActivationFunction = cfELM.get('activationFunction', columns=['activationFunction'])
            self.elmFPActivationFunction = confDictActivationFunctions[self.elmFPActivationFunction['activationFunction']]
            self.elmFPNumberNeurons = self.elmFPInputWeights.shape[0]
            self.elmFPTrained = True
            cfELM = pycassa.ColumnFamily(self.pool, 'elmFN')
            self.elmFNInputWeights = self._cassandraToNumpyArray(cfELM, 'inputWeights')
            self.elmFNInputBiases = self._cassandraToNumpyArray(cfELM, 'inputBiases')
            self.elmFNOutputWeights = self._cassandraToNumpyArray(cfELM, 'outputWeights')
            self.elmFNActivationFunction = cfELM.get('activationFunction', columns=['activationFunction'])
            self.elmFNActivationFunction = confDictActivationFunctions[self.elmFNActivationFunction['activationFunction']]
            self.elmFNNumberNeurons = self.elmFNInputWeights.shape[0]
            self.elmFNTrained = True
            if __debug__: print 'Found ELM models in database.'
        except pycassa.NotFoundException:
            # If we cannot find some of the required data for the ELM, we train it
            # and store to Cassandra
            print 'No ELM data found in database, proceeding to training.'
            self._elmTrain(optimizeFP=True)
            self._elmTrain(optimizeFP=False)
            if __debug__: print 'Trying for ELM models in database...'
            cfELM = pycassa.ColumnFamily(self.pool, 'elmFP')
            self.elmFPInputWeights = self._cassandraToNumpyArray(cfELM, 'inputWeights')
            self.elmFPInputBiases = self._cassandraToNumpyArray(cfELM, 'inputBiases')
            self.elmFPOutputWeights = self._cassandraToNumpyArray(cfELM, 'outputWeights')
            self.elmFPActivationFunction = cfELM.get('activationFunction', columns=['activationFunction'])
            self.elmFPActivationFunction = confDictActivationFunctions[self.elmFPActivationFunction['activationFunction']]
            self.elmFPNumberNeurons = self.elmInputWeights.shape[0]
            self.elmFPTrained = True
            cfELM = pycassa.ColumnFamily(self.pool, 'elmFN')
            self.elmFNInputWeights = self._cassandraToNumpyArray(cfELM, 'inputWeights')
            self.elmFNInputBiases = self._cassandraToNumpyArray(cfELM, 'inputBiases')
            self.elmFNOutputWeights = self._cassandraToNumpyArray(cfELM, 'outputWeights')
            self.elmFNActivationFunction = cfELM.get('activationFunction', columns=['activationFunction'])
            self.elmFNActivationFunction = confDictActivationFunctions[self.elmFNActivationFunction['activationFunction']]
            self.elmFNNumberNeurons = self.elmInputWeights.shape[0]
            self.elmFNTrained = True
            if __debug__: print 'Found ELM models in database.'


    def run(self, gemmyTask):
        """
            :rtype: dict of format:
      
            .. code-block:: javascript
                
                { 'verdict': str, 
                  'confidence': float, 
                  'parameters': 
                                { 'keySpace': str, 
                                  'serverList': list, 
                                  'poolSize': int, 
                                  'timeOut': float or ``None``, 
                                  'maxNumberHashes': int, 
                                  'usedFeatures': list, 
                                  'maxNumberNeighbors': int, 
                                  'maxRefSamplesCheckedAgainst': int
                                }
                }

            Executes the Gemmy algorithm for the given *gemmyTask* of type :class:`gemmy.GemmyTask`
        """
        if not isinstance(gemmyTask, GemmyTask):
            raise Exception('Given task is not an instance of GemmyTask (gotten '+type(gemmyTask)+' ).')
        self.task = gemmyTask
        self.minHashDict = {}
        for featureNumber in self.task.pp_results.keys():
            self.minHashDict[featureNumber] = []
            for featureValue in self.task.pp_results[featureNumber]:
                if ',' in featureValue:
                    featureValue = featureValue.split(',')[1]
                if featureValue not in self.minHashDict[featureNumber]:
                    if self.minHashDict[featureNumber].__len__()+1 > self.maxNumberHashes:
                        if self.minHashDict[featureNumber][-1] > featureValue:
                            self.minHashDict[featureNumber].pop()
                            bisect.insort(self.minHashDict[featureNumber], featureValue)
                    else:
                        bisect.insort(self.minHashDict[featureNumber], featureValue)
        if self.task.isReference():
            SMAFact = self._insertReferenceSampleInCassandra()
        else:
            SMAFact = self._getDecision()

        SMAFact['parameters'] = {'keySpace': self.keySpace,\
                                 'serverList': self.serverList,\
                                 'poolSize': self.poolSize,\
                                 'timeOut': self.timeOut,\
                                 'maxNumberHashes': self.maxNumberHashes,\
                                 'usedFeatures': self.usedFeatures,\
                                 'maxNumberNeighbors': self.maxNumberNeighbors,\
                                 'maxRefSamplesCheckedAgainst': self.maxRefSamplesCheckedAgainst
                                 }

        
        return SMAFact


    def _runTaskForTraining(self, gemmyTask):
        """
            Executes the Gemmy algorithm for the given *gemmyTask* of type :class:`gemmy.GemmyTask`
        """
        if not isinstance(gemmyTask, GemmyTask):
            raise Exception('Given task is not an instance of GemmyTask (gotten '+type(gemmyTask)+' ).')
        self.task = gemmyTask
        self.minHashDict = {}
        for featureNumber in self.task.pp_results.keys():
            self.minHashDict[featureNumber] = []
            for featureValue in self.task.pp_results[featureNumber]:
                if ',' in featureValue:
                    featureValue = featureValue.split(',')[1]
                if featureValue not in self.minHashDict[featureNumber]:
                    if self.minHashDict[featureNumber].__len__()+1 > self.maxNumberHashes:
                        if self.minHashDict[featureNumber][-1] > featureValue:
                            self.minHashDict[featureNumber].pop()
                            bisect.insort(self.minHashDict[featureNumber], featureValue)
                    else:
                        bisect.insort(self.minHashDict[featureNumber], featureValue)
        self._getNeighbors()


    def _insertReferenceSampleInCassandra(self):
        """
            :rtype: dict of format:

            .. code-block:: javascript
                
                { 'verdict': 'reference', 
                  'confidence': 1.0, 
                  'parameters': 
                                { 'keySpace': str, 
                                  'serverList': list, 
                                  'poolSize': int, 
                                  'timeOut': float or ``None``, 
                                  'maxNumberHashes': int, 
                                  'usedFeatures': list, 
                                  'maxNumberNeighbors': int, 
                                  'maxRefSamplesCheckedAgainst': int
                                }
                }

            Creates an entry for this file in the Cassandra Column Families.
            Stores the MinHash data in the *minhashdata* CF, and sample \
            statistics (i.e. metadata and amount of hashes per feature) in the *featurestats* CF.

        """ 

        # Get a hold of the Column Families used for the insertion
        cfMinhashdata = pycassa.ColumnFamily(self.pool, 'minhashdata')
        cfFeaturestats = pycassa.ColumnFamily(self.pool, 'featurestats')

        # Insert the statistics about the file in the CF: For now just the number of 
        # features for each feature number. Speeds up the Jaccard calculation by
        # avoiding counting at each Jaccard calculation request
        with cfFeaturestats.batch(queue_size=7) as cfFeaturestatsBatch:
            cfFeaturestatsBatch.insert(self.task.sha1, dict([(str(feature),str(length)) \
                                                for feature, length in zip(self.minHashDict.keys(),[self.minHashDict[key].__len__()\
                                                for key in self.minHashDict.keys()])]))
            cfFeaturestatsBatch.insert(self.task.sha1, {'system': str(self.task.system)})
            cfFeaturestatsBatch.insert(self.task.sha1, {'classification': str(self.task.classification)})
            cfFeaturestatsBatch.insert(self.task.sha1, {'first_seen': str(self.task.first_seen)})
            cfFeaturestatsBatch.insert(self.task.sha1, {'sha256': str(self.task.sha256)})
            cfFeaturestatsBatch.insert(self.task.sha1, {'md5': str(self.task.md5)})
            cfFeaturestatsBatch.insert(self.task.sha1, {'size': str(self.task.size)})


        # Perform the insertion of the MinHash data in the CF. Done feature per feature
        # The send order is implicitly sent when exiting the with clause
        with cfMinhashdata.batch(queue_size=10) as cfMinhashdataBatch:
            for feature in self.minHashDict.keys():
                cfMinhashdataBatch.insert(str(self.task.sha1)+':'+str(feature), dict([(hash, '1') for hash in self.minHashDict[feature]]))
                  
        SMAFact = {'verdict': 'reference', 'confidence': 1.0}
        return SMAFact

    def _getDecision(self):
        """
            Calls :func:`_getNeighbors` to fetch the neighbors of the sample, \
            and uses them to make an ensemble decision using :func:`_takeEnsembleDecision`.

        """
        self._getNeighbors()
        # If we get no nearest neighbors, it means the restrictions on the
        # features to use for comparison where too strict and the sample
        # does not have these features. Or that the reference set is too small/restrictive.
        if self.neighbors.__len__() == 0:
            verdict = 'clean'
            confidence = 0.0
        else:
            verdict, confidence = self._takeEnsembleDecision()
        
        SMAFact = {'verdict': verdict, 'confidence': confidence}
        return SMAFact

    ################### ELM Functions Section ###################

    def _numpyArrayToCassandra(self, numpyArray, columnFamily, key):
        """ Puts a Numpy array in a Cassandra key. Uses Pickle to 
            store the matrix to a string and back.
        """
        if __debug__: print 'Serializing matrix and storing in database.'
        if not isinstance(columnFamily, pycassa.ColumnFamily):
            raise Exception('Given Column Family is not an instance of a Column Family.')
        if not isinstance(numpyArray, np.ndarray):
            raise Exception('Given array os not a Numpy Array.')
        if not isinstance(key, str):
            raise Exception('Given key is not a string.')
        with columnFamily.batch(queue_size=2) as columnFamilyBatch:
            columnFamilyBatch.insert(key, {'values': Pickle.dumps(numpyArray.ravel())})
            columnFamilyBatch.insert(key, {'shape': Pickle.dumps(numpyArray.shape)})


    def _cassandraToNumpyArray(self, columnFamily, key):
        """ Returns a previously stored Numpy array from Cassandra.
        """
        if __debug__: print 'De-serializing matrix and to Numpy array.'
        if not isinstance(columnFamily, pycassa.ColumnFamily):
            raise Exception('Given Column Family is not an instance of a Column Family.')
        if not isinstance(key, str):
            raise Exception('Given key is not a string.')
        data = columnFamily.get(key, columns=['shape', 'values'])
        numpyArray = Pickle.loads(data['values']).reshape(Pickle.loads(data['shape']))
        return numpyArray


    def _elmGetNearestNeighborOppositeClass(self, neighbors, classif):
        if __debug__: print 'Getting nearest neighbor of opposite class...'
        if classif == 1:
            # We are looking for the nearest clean sample
            oppClassif = 'clean'
        else:
            # We look for the nearest malware
            oppClassif = 'malware'
        listNeighbors = [item[0] for item in neighbors]
       
        cfFeaturestats = pycassa.ColumnFamily(self.pool, 'featurestats')
        tempListClasses = cfFeaturestats.multiget(listNeighbors, columns=['system'])
        rank, distance = [(i,item2[1]) for i,item2 in enumerate(neighbors) if tempListClasses[item2[0]]['system']==oppClassif][0]
        #rank, distance = [(i,neighbors[i][1]) for i, item2 in enumerate([item[1] for item in neighbors if item[1]==oppClassif])][0]
        if rank==0:
            rank = confMaxRefSamplesCheckedAgainst
            distance = 1.0
        if __debug__: print 'Found number '+str(rank)+' at distance '+str(distance)
        return rank, distance


    def _elmGetNearestNeighbor(self, neighbors):
        if __debug__: print 'Getting the nearest neighbor...'
        refFileHash = neighbors[1][0]
        cfFeaturestats = pycassa.ColumnFamily(self.pool, 'featurestats')
        verdict = cfFeaturestats.get(refFileHash, columns=['classification']).values()[0]
        if verdict=='clean':
            verdict=-1
        else:
            verdict=1
        distance = neighbors[1][1]
        if __debug__: print 'Nearest neighbor is '+str(verdict)+' at distance '+str(distance)
        return verdict, distance

    def _elmGenerateTaskFromHash(self, hash):
        JSONData = {"priority":"", "sha1":"", "preprocessor_data": {}, "metadata":{}}   

        JSONData["priority"] = 0
        JSONData["sha1"] = hash
        JSONData["preprocessor_data"] = { "classification_status":"unknown",\
                                          "pp_results":{}\
                                        }
        JSONData["metadata"] = { "sha1":"",\
                                 "system":"",\
                                 "first_seen":"",\
                                 "sha256":"",\
                                 "md5":"",\
                                 "size":0}  
        
        cfFeaturestats = pycassa.ColumnFamily(self.pool, 'featurestats')
        cfMinhashdata = pycassa.ColumnFamily(self.pool, 'minhashdata')
        allFeatureStatsData = cfFeaturestats.get(hash)
        listFeatures = []
        for key in allFeatureStatsData.keys():
            if key in ['classification', 'first_seen', 'md5', 'sha256', 'size', 'system']:
                if key == "first_seen":
                    JSONData["metadata"]["first_seen"] = allFeatureStatsData['first_seen']
                if key == "md5":
                    JSONData["metadata"]["md5"] = allFeatureStatsData['md5']
                if key == "sha256":
                    JSONData["metadata"]["sha1"] = allFeatureStatsData['sha256']
                    JSONData["metadata"]["sha256"] = allFeatureStatsData['sha256']
                if key == "size":
                    JSONData["metadata"]["size"] = int(allFeatureStatsData['size'])
                if key == "system":
                    JSONData["metadata"]["system"] = allFeatureStatsData['system']
                    JSONData["metadata"]["classification"] = allFeatureStatsData['system']
            else:
                listFeatures.append(key)

        for featureNumber in listFeatures:
            featureValues = cfMinhashdata.get(hash+':'+featureNumber, column_count=confRefSamplesMaxNumberHashes)
            featureValues = featureValues.keys()
            JSONData["preprocessor_data"]["pp_results"][str(featureNumber)]=featureValues  

        task = GemmyTask(JSONData)
        
        return task


    def _elmCreateTrainingData(self, listHashes):
        trainDataInput = np.zeros((len(listHashes), 3))
        trainDataOutput = np.zeros((len(listHashes), 1))
        if __debug__: print 'Creating Training Data...'
        for index, hash in enumerate(listHashes):
            task = self._elmGenerateTaskFromHash(hash)
            self._runTaskForTraining(task)
            neighbors = self.neighbors
            classif1NN, distance1NN = self._elmGetNearestNeighbor(neighbors)
            rankKNN, distanceKNN = self._elmGetNearestNeighborOppositeClass(neighbors, classif1NN)
            trainDataInput[index, :] = [float(distance1NN), float(rankKNN), float(distanceKNN)]
            trainDataOutput[index, 0] = float(classif1NN)
        return trainDataInput, trainDataOutput
            


    def _elmTrain(self, numSamples=confELMNumSamples, \
                        numberNeurons=confELMNumberNeurons, \
                        activationFunction=confELMActivationFunction, \
                        optimizeFP=True):
        if not isinstance(numSamples, int):
            raise Exception('Given number of samples for training is not an integer.')
        self.elmActivationFunction = confDictActivationFunctions[activationFunction]
        self.elmNumberNeurons = numberNeurons
        # MORE CHECKING HERE

        # Get the list of Reference samples to use for the Training
        listHashes = self.refFilesHashes[:numSamples]
        elmTrainDataInput, elmTrainDataOutput = self._elmCreateTrainingData(listHashes)
        if __debug__: print 'Gotten Training Data, training ELM...'
        elmTrainDataInputDimensionality = elmTrainDataInput.shape[1]

        # Train here:
        self.elmInputWeights = np.random.random((self.elmNumberNeurons, elmTrainDataInputDimensionality))*2-1
        self.elmInputBiases = np.random.random((self.elmNumberNeurons, 1))
        self.elmHiddenLayer = np.dot(elmTrainDataInput, self.elmInputWeights.T)
        self.elmHiddenLayer += np.tile(self.elmInputBiases, (1, self.elmHiddenLayer.shape[0])).T
        self.elmHiddenLayer = self.elmActivationFunction(self.elmHiddenLayer)
        self.elmOutputWeights = np.dot(np.linalg.pinv(self.elmHiddenLayer), elmTrainDataOutput)
        self.elmTrainOutput = np.dot(self.elmHiddenLayer, self.elmOutputWeights)
        self.elmTrainError = np.mean(np.power(self.elmTrainOutput-elmTrainDataOutput, 2), axis=0)

        if __debug__: print 'ELM trained, sending to Cassandra...'
        # Once trained, push it to Cassandra
        if optimizeFP:
            nameCF = 'elmFP'
        else:
            nameCF = 'elmFN'
        cfELM = pycassa.ColumnFamily(self.pool, nameCF)
        self._numpyArrayToCassandra(self.elmInputWeights, cfELM, 'inputWeights')
        self._numpyArrayToCassandra(self.elmInputBiases, cfELM, 'inputBiases')
        self._numpyArrayToCassandra(self.elmOutputWeights, cfELM, 'outputWeights')
        cfELM.insert('activationFunction', {'activationFunction': confDictActivationFunctionsReverse[self.elmActivationFunction]})
        self._numpyArrayToCassandra(self.elmInputWeights, cfELM, 'inputWeights')

        if __debug__: print 'ELM Trained with '+str(self.elmNumberNeurons)+' neurons, giving MSE: '+str(self.elmTrainError)


    def _elmTest(self, verdict1NN, thresholdClassifier):
        verdict1NNTest, distance1NNTest = self._elmGetNearestNeighbor(self.neighbors)
        rankNNOp, distanceNNOpp = self._elmGetNearestNeighborOppositeClass(self.neighbors, verdict1NNTest)
        elmTestData = np.array([distance1NNTest,rankNNOp,distanceNNOpp])
        if verdict1NN == 'clean':
            hiddenLayerTest = np.dot(elmTestData, self.elmFNInputWeights.T).reshape((1, self.elmFNNumberNeurons))
            hiddenLayerTest += np.tile(self.elmFNInputBiases, (1, hiddenLayerTest.shape[0])).T
            hiddenLayerTest = self.elmFNActivationFunction(hiddenLayerTest)
            distance = np.dot(hiddenLayerTest, self.elmFNOutputWeights)[0]
        else:
            hiddenLayerTest = np.dot(elmTestData, self.elmFPInputWeights.T).reshape((1, self.elmFPNumberNeurons))
            hiddenLayerTest += np.tile(self.elmFPInputBiases, (1, hiddenLayerTest.shape[0])).T
            hiddenLayerTest = self.elmFPActivationFunction(hiddenLayerTest)
            distance = np.dot(hiddenLayerTest, self.elmFPOutputWeights)[0][0]
        if distance > thresholdClassifier:
            verdict = verdict1NN
        else:
            verdict = 'unknown'
        return verdict, distance

    ###################### END OF ELM FUNCTION SECTION ###################

    def _takeEnsembleDecision(self):
        """
            :rtype: couple ('verdict', confidence) with 'verdict' a *str* and confidence a *float*

            Takes a decision on the current sample.
        """ 

        self.verdict1, self.confidence1 = self._takeDecisionClassifier1()
        self.verdict2, self.confidence2 = self._takeDecisionClassifier2(self.verdict1)

        if __debug__: print '1NN says '+str(self.verdict1)+' with confidence '+str(100*float(self.confidence1))+'%.'
        if __debug__: print 'ELM says '+str(self.verdict2)+' with confidence '+str(100*float(self.confidence2))+'%.'
        
        return self.verdict2, self.confidence2


    def _takeDecisionClassifier1(self, thresholdClassifier=confThresholdClassifier1):
        """ 
            :param thresholdClassifier: Threshold above which the classifier says 'unknown'
            :type thresholdClassifier: float [defaults to *confThresholdClassifier1*]
            :rtype: couple ('verdict', confidence) with 'verdict' a *str* and confidence a *float*

            Takes a decision on the current sample based on Classifier1's strategy: 
            
            * Look at the first Nearest Neighbor
                * The verdict is that of the nearest neighbor in question
            
            This strategy is based on the report from Petteri Hyvarinen showing (rather unusually) \
            that taking the first nearest neighbor only to make a decision on the sample at hand is \
            always a better idea than taking more (and following the usual KNN majority vote principle).    

        """
        nearestNeighbor, distance = self.neighbors[0]
        if distance > thresholdClassifier:
            verdict = 'clean'
            confidence = 0.0
        else:
            
            cfFeaturestats = pycassa.ColumnFamily(self.pool, 'featurestats')
            refFileHash = self.neighbors[0][0]
            verdict = cfFeaturestats.get(refFileHash, columns=['classification']).values()[0]
            if verdict in classificationNamesClean:
                verdict = 'clean'
                confidence = 1.0-distance
            elif verdict in classificationNamesMalware:
                verdict = 'malware'
                confidence = 1.0-distance
            else:
                verdict = 'clean'
                confidence = 0.0
        return verdict, confidence


    def _takeDecisionClassifier2(self, firstStepVerdict, thresholdClassifier=confThresholdClassifier2):
        """ 
            :param firstStepVerdict: Verdict obtained by the 1-NN
            :type thresholdClassifier: float [defaults to *confThresholdClassifier2*]
            :rtype: couple ('verdict', confidence) with 'verdict' a *str* and confidence a *float*

            Takes a final decision on the current sample based on Classifier1's verdict, using ELM.  

        """
        verdict, confidence = self._elmTest(firstStepVerdict, thresholdClassifier)
        return verdict, confidence



    def _getNeighbors(self): 
        """ 
            Computes the minHash estimation of the Jaccard using Cassandra backend \
            and gives the maxNumberNeighbors nearest neighbors and their distance in a list of couples \
            stored in the object's internal variable *neighbors*. The overall algorithm goes:

            #. Get a hold of the Column Families used for the Nearest Neighbor using Jaccard
            #. Compare the features requested against the available ones
            #. If there are no common features in any of the 2 files, return empty list of neighbors
            #. Initialize list of all pairwise distances to zero
            #. Iterate through each available/requested feature to calculate the Jaccard:
         
                * Technically, it computes this (with capitals being ensembles and :math:`|X|` denoting\
                  cardinality of ensemble :math:`X`):

                    .. math::`J(A,B) = \\frac{1}{|C|}\\sum_{i \in C}\\frac{|A_{i}\\cap B_{i}|}{|A_{i}|+|B_{i}|-|A_{i}\\cap B_{i}|}`

                  Where :math:`A_{i}` and :math:`B_{i}` are the sets of feature values for feature number\
                  :math:`i` for file :math:`A` and :math:`B` resp.,\
                  and :math:`C = \\left(Feature Numbers of A\\right)\\bigcup\\left(Feature Numbers of B\\right)`
                * The sets are reduced to their MinHash, here, compared to the above formula
                * Separate the calculation in getting first the numerators, i.e. the intersection
                * Denominators are a bit tedious to get in a one-liner: Uses the featurestats CF and\
                  previous calculation of the intersection, the numerator
                * Divide and cumulative add, see formula above

            #. Normalize and convert the similarity to a distance. Sort to get the closest neighbors first

        """ 

        if __debug__: timerBegin = time.time()

        # Get a hold of the Column Families used for the Nearest Neighbor using Jaccard
        cfMinhashdata = pycassa.ColumnFamily(self.pool, 'minhashdata')
        cfFeaturestats = pycassa.ColumnFamily(self.pool, 'featurestats')    

        # Compare the features requested against the available ones
        if self.usedFeatures.__len__() == 0:
            usableFeatures = self.minHashDict.keys()
        else:
            usableFeatures = [item for item in self.usedFeatures if item in self.minHashDict.keys()] 

        # If there are no common features in any of the 2 files, return empty list of neighbors
        if usableFeatures.__len__() == 0:
            self.neighbors = []
            return
            
        # Initialize list of all pairwise distances to zero. Distance from sample to each of the 
        # MAX from allTrainFileNames will be held in this
        minHashDistances = [0.0]*self.refFilesHashes.__len__()    

        # Iterate through each available/requested features to calculate the Jaccard
        # Technically, it computes this (with capitals being ensembles and |X| denoting
        # cardinality of ensemble X):
        # J(A,B) = \frac{1}{|C|}\sum_{i \in C}\frac{|A_{i}\cap B_{i}|}{|A_{i}|+|B_{i}|-|A_{i}\cap B_{i}|}
        # Where A_{i} and B_{i} are the sets of feature values for feature number i for file A and B resp.,
        # and C = \left(Feature Numbers of A\right)\bigcup\left(Feature Numbers of B\right) 
        # The sets are reduced to their MinHash, here, compared to the above formula
        for feature in usableFeatures:
            # Separate the calculation in getting first the numerators, i.e. the intersection
            tempListNumerators = None
            while tempListNumerators is None:
                try:
                    tempListNumerators = cfMinhashdata.multiget_count([item+':'+str(feature)\
                                                                       for item in self.refFilesHashes], \
                                                                       columns=self.minHashDict[feature])
                except pycassa.pool.AllServersUnavailable:
                    print 'No Cassandra server responding, retrying in 5 seconds...'
                    time.sleep(5)   

            numerators = [tempListNumerators[fileHash+':'+feature]\
                            if tempListNumerators.has_key(fileHash+':'+feature)\
                            else 0.0\
                            for fileHash in self.refFilesHashes]  

            # Denominators are a bit tedious to get in a one-liner: Uses the featurestats CF and 
            # previous calculation of the intersection, the numerator
            tempListNumFeatureValues = None
            while tempListNumFeatureValues is None:
                try:
                    tempListNumFeatureValues = cfFeaturestats.multiget(self.refFilesHashes, columns=[str(feature)])
                except pycassa.pool.AllServersUnavailable:
                    print 'No Cassandra server responding, retrying in 5 seconds...'
                    time.sleep(5)
            for fileHash in tempListNumFeatureValues.keys():
                tempListNumFeatureValues[fileHash][feature] = min(float(tempListNumFeatureValues[fileHash][feature]), self.maxNumberHashes)    

            denominators = [numFeaturesTestFile+numFeaturesTrainFile-numFeaturesIntersection for \
                              numFeaturesTestFile, numFeaturesTrainFile,numFeaturesIntersection in \
                              zip([self.minHashDict[feature].__len__()]*numerators.__len__(), \
                                  [float(tempListNumFeatureValues[fileHash][feature])\
                                    if tempListNumFeatureValues.has_key(fileHash)\
                                    else 0.0\
                                    for fileHash in self.refFilesHashes],\
                                 numerators)]   
            # Divide and cumulative add, see formula above
            minHashDistances = [previous+float(numerator)/float(denominator) for previous, numerator, denominator in\
                                                                        zip(minHashDistances, numerators, denominators)]    

        # Normalize and convert the similarity to a distance. Sort to get the closest neighbors
        minHashDistances = [1.0-previous/usableFeatures.__len__() for previous in minHashDistances]
        self.neighbors = zip(self.refFilesHashes, minHashDistances)
        self.neighbors.sort(key=lambda minHashDistances:minHashDistances[1]) 

        if __debug__: print 'Elapsed time for this sample: '+str(time.time()-timerBegin)+'s.'

