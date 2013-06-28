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


    def _takeEnsembleDecision(self):
        """
            :rtype: couple ('verdict', confidence) with 'verdict' a *str* and confidence a *float*

            Takes a decision on the current sample.
            This is where we make the ensemble if any, based on the decisions \
            provided by the different classifiers.  

        """ 

        verdict1, confidence1 = self._takeDecisionClassifier1()

        print 'Classifier1 says '+str(verdict1)+' with confidence '+str(100*float(confidence1))+'%.'
        
        return verdict1, confidence1


    def _takeDecisionClassifier1(self, thresholdClassifier=0.9999):
        """ 
            :param thresholdClassifier: Threshold above which the classifier says 'unknown'
            :type thresholdClassifier: float [defaults to *confThresholdClassifier1*]
            :rtype: couple ('verdict', confidence) with 'verdict' a *str* and confidence a *float*

            Takes a decision on the current sample based on Classifier1's strategy: 
            
            * Look at the first Nearest Neighbor
            * If the distance to this neighor is greater than a predefined threshold:
                * Decide that we do not know (verdict is 'unknown'), to limit false positives
            * Else:
                * The verdict is that of the nearest neighbor in question
            
            This strategy is based on the report from Petteri Hyvarinen showing (rather unusually) \
            that taking the first nearest neighbor only to make a decision on the sample at hand is \
            always a better idea than taking more (and following the usual KNN majority vote principle).    

            This strategy gives a rather good coverage of the data, when having reasonable threshold.
            The threshold gives direct control on the Coverage/False Positives tradeoff:
         
                * High threshold means good coverage but high false positives rate
                * Low threshold means bad coverage but low false positives rate 

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









