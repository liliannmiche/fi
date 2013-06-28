"""
Task handlers for the consumer class.
Basically a collection of functions which execute depending on the job's tube.

@author: Yoan Miche
@email: yoan.miche@aalto.fi
@version: 1.0
"""



""" WORK IN PROGRESS """

# Global imports
import numpy as np
import os
import bisect

# Cassandra API
import pycassa

# Local imports
from cassandraconnection import *


def minHashCassandra(job):
    """ Computes the minHash for a job using Cassandra backend:
            - Computes a hash of each feature value pair (currently identity function)
            - Sorts the hashes per value
            - Writes them to an entry in Cassandra's CF 'minhashdata'
    """

    # Define the hash function to use
    # Should be Rabin fingerprinting function to respect some minhash properties
    # identity function for now, since values are MD5/SHA1hashes
    hashFunction = lambda x : x

    fileName = job.body
    
    createMinHashEntryCassandra(fileName, hashFunction)
    
    job.sequence.pop()
    return job


def createMinHashEntryCassandra(fileName,\
                                                        hashFunction,\
                                                        MAX_NUMBER_HASHES=100):
    """ Creates an entry for this file in the Cassandra minhash column family.
    """

    # Connect and get a connection pool. Size should be something more appropriate, but what?
    pool = pycassa.pool.ConnectionPool(cassandraKeySpace, \
                                                               server_list=cassandraHostAndPort,\
                                                               pool_size=5,\
                                                               timeout=None)

    # Get a hold of the Column Families used for the insertion
    cfMinhashdata = pycassa.ColumnFamily(pool, 'minhashdata')
    cfFilelist = pycassa.ColumnFamily(pool, 'filelist')
    cfFeaturestats = pycassa.ColumnFamily(pool, 'featurestats')

    # Insert this filename in the list of available filenames held in Cassandra
    cfFilelist.insert('filelist', {fileName: '1'})

    # Create a dictionary holding the feature values for each feature number
    # Only keep the smallest MAX ones in the dictionary (per feature number)
    # Bisect and pop allow this insertion in place
    file = open(fileName, 'r')
    hashedValuesDict = {}
    for line in file.readlines():
        featureNumber, featureValue = line.split(' ')
        if not hashedValuesDict.has_key(featureNumber):
            hashedValuesDict[featureNumber] = []
        if ',' in featureValue:
            featureValue = featureValue.split(',')[1]
        featureValue = featureValue.rstrip()
        if hashedValuesDict[featureNumber].__len__()+1 > MAX_NUMBER_HASHES:
            if hashedValuesDict[featureNumber][-1] > featureValue:
                hashedValuesDict[featureNumber].pop()
                bisect.insort(hashedValuesDict[featureNumber], featureValue)
        else:
            bisect.insort(hashedValuesDict[featureNumber], featureValue)

    
    # Insert the statistics about the file in the CF: For now just the number of 
    # features for each feature number. Speeds up the Jaccard calculation by
    # avoiding counting at each Jaccard calculation request
    cfFeaturestats.insert(fileName, dict([(str(feature),str(length)) \
                                                              for feature, length in zip(hashedValuesDict.keys(),[hashedValuesDict[key].__len__()\
                                                              for key in hashedValuesDict.keys()])]))

    # Perform the insertion of the MinHash data in the CF. Done feature per feature.
    # Possible to do it all in one shot? Worth it?
    for feature in hashedValuesDict.keys():
        cfMinhashdata.insert(fileName+':'+str(feature), dict([(hash, '1') for hash in hashedValuesDict[feature]]))

    # Clean before leaving
    pool.dispose()
    file.close()



def minHashTestCassandra(job):
    """ Computes the minHash for a job using Cassandra backend:
    """

    # Define the hash function to use
    # Should be Rabin fingerprinting function to respect some minhash properties
    # identity function for now, since values are MD5/SHA1hashes
    hashFunction = lambda x : x

    fileName = job.body
    
    minHashNNCassandra(fileName, hashFunction)
    
    job.sequence.pop()
    return job



def writeNeighborsFile(testFileName, MAX_NUMBER_NEIGHBORS, allTrainFileNames, sortedIndices, minHashDistances):
    """ Helper function to write the result of the neighbors query to a file.
         Overwrites any existing file.
    """
    neighborsFileName = testFileName.rsplit('.')[0]+'.all-neighbors-ref-50k-h-50-f-all'
    neighborsFile = open(neighborsFileName, 'w')
    for i in range(MAX_NUMBER_NEIGHBORS):
        neighborsFile.write(str(allTrainFileNames[sortedIndices[i]])+' '+str(minHashDistances[sortedIndices[i]])+'\n')
    neighborsFile.close()


def minHashNNCassandra(testFileName, hashFunction,\
                                           usedFeatures=[], MAX_NUMBER_NEIGHBORS=49999,\
                                           MAX_NUMBER_HASHES=50,\
                                           MAX_REF_FILES_CHECKED = 50000):

    """ Computes the minHash estimation of the Jaccard using Cassandra backend and gives the
         MAX_NUMBER_NEIGHBORS nearest neighbors in the .neighbors file.
    """

    # Connect and get a connection pool. Size should be something more appropriate, but what?
    pool = pycassa.pool.ConnectionPool(cassandraKeySpace,\
                                                                server_list=cassandraHostAndPort,\
                                                                pool_size=5, timeout=None)

    # Get a hold of the Column Families used for the Nearest Neighbor using Jaccard
    cfMinhashdata = pycassa.ColumnFamily(pool, 'minhashdata')
    cfFilelist = pycassa.ColumnFamily(pool, 'filelist')
    cfFeaturestats = pycassa.ColumnFamily(pool, 'featurestats')

    # Get the list of all the reference files in the base to compare to
    # Limited by MAX. Since files have "random" filenames, the MAX
    # selected ones are rather taken at random
    allTrainFileNames = cfFilelist.get(key='filelist', column_count=MAX_REF_FILES_CHECKED).keys()

    # Create a dictionary holding the feature values for each feature number
    # Only keep the smallest MAX ones in the dictionary (per feature number)
    # Bisect and pop allow this insertion in place
    hashedValuesDict = {}
    testFile = open(testFileName, 'r')
    for line in testFile.readlines():
        featureNumber, featureValue = line.split(' ')
        if featureNumber not in hashedValuesDict:
            hashedValuesDict[featureNumber] = []
        if ',' in featureValue:
            featureValue = featureValue.split(',')[1]
        featureValue = featureValue.rstrip()
        if hashedValuesDict[featureNumber].__len__()+1 > MAX_NUMBER_HASHES:
            if hashedValuesDict[featureNumber][-1] > featureValue:
                hashedValuesDict[featureNumber].pop()
                bisect.insort(hashedValuesDict[featureNumber], featureValue)
        else:
            bisect.insort(hashedValuesDict[featureNumber], featureValue)

    # Compare the features requested against the available ones
    if usedFeatures.__len__() == 0:
        usableFeatures = hashedValuesDict.keys()
    else:
        usableFeatures = [item for item in usedFeatures if item in hashedValuesDict.keys()]

    # If there are no common features in any of the 2 files, return max distance
    if usableFeatures.__len__() == 0:
        return 1.0
    
    # Initialize list of all pairwise distances to zero. Distance from testFile to each of the 
    # MAX from allTrainFileNames will be held in this
    minHashDistances = np.zeros((allTrainFileNames.__len__(),))

    # Iterate through each available/requested features to calculate the Jaccard
    # Technically, it computes this (with capitals being ensembles and |X| denoting
    # cardinality of ensemble X):
    # J(A,B) = \frac{1}{|C|}\sum_{i \in C}\frac{|A_{i}\cap B_{i}|}{|A_{i}|+|B_{i}|-|A_{i}\cap B_{i}|}
    # Where A_{i} and B_{i} are the sets of feature values for feature number i for file A and B resp.,
    # and C = \left(Feature Numbers of A\right)\bigcup\left(Feature Numbers of B\right) 
    # The sets are reduced to their MinHash, here, compared to the above formula
    for feature in usableFeatures:
        # Separate the calculation in getting first the numerators, i.e. the intersection
        numerators = cfMinhashdata.multiget_count([item+':'+str(feature) for item in allTrainFileNames], \
                                                                                             columns=hashedValuesDict[feature]).values()

        # Denominators are a bit tedious to get in a one-liner: Uses the featurestats CF and 
        # previous calculation of the intersection, the numerator
        denominators = [numFeaturesTestFile+numFeaturesTrainFile-numFeaturesIntersection for \
                                    numFeaturesTestFile, numFeaturesTrainFile,numFeaturesIntersection in \
                                    zip([hashedValuesDict[feature].__len__()]*numerators.__len__(), \
                                         [int(item.values()[0]) for item in cfFeaturestats.multiget(allTrainFileNames,\
                                         columns=[str(feature)]).values()], numerators)]

        # Divide and cumulative add, see formula above
        minHashDistances += np.array([float(numerator)/denominator\
                                                            for numerator, denominator in zip(numerators, denominators)])

    # Normalize and convert the similarity to a distance. Sort to get the closest neighbors
    minHashDistances = (np.ones((allTrainFileNames.__len__(),))-minHashDistances/usableFeatures.__len__())
    sortedIndices = minHashDistances.argsort()

    # Write results to file
    writeNeighborsFile(testFileName, MAX_NUMBER_NEIGHBORS, allTrainFileNames, sortedIndices, minHashDistances)

    pool.dispose()
    testFile.close()




taskHandlers = { minHashCassandra:        ['mhc','minhashcassandra','minHashCassandra'],
                            minHashTestCassandra:  ['mhtc','minhashtestcassandra']}



# Helper functions
def findKey(dictionary, value):
    """ Returns the key of dictionary given a value.
        Only gets the first key ! 
    """
    return [key for key, values in dictionary.iteritems() if value in values][0]



def getTaskHandler(taskHandler):
    """ Returns the right task handler name based on the task name given.
        Raises exception if no handler name is close enough.
    """
    if taskHandler in [i for s in taskHandlers.values() for i in s]:
        return findKey(taskHandlers, taskHandler)
    else:
        raise ValueError('Given task handler name does not correspond to an existing implementation.')
