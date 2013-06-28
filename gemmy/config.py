"""
Algorithms parameters and Connection parameters for Cassandra database.


@author: Yoan Miche
@email: yoan.miche@aalto.fi
@version: 1.0
"""

# @TODO :
#           - Check the Cassandra parameters (i.e. TimeOut and PoolSize)
#			- Update the Classifier parameters once strategy is finalized

import numpy as np

############### Cassandra parameters #################
confCassandraKeySpace = 'jozsef'
confCassandraHosts = ['130.233.178.108','130.233.178.110','130.233.178.55']
confCassandraPort = 9160
confCassandraHostAndPort = [item+':'+str(confCassandraPort) for item in confCassandraHosts]
confCassandraTimeOut = None # This to avoid timeouts due to requests taking long time to execute.
confCassandraPoolSize = 5 # No reason for this value and no idea what it should/could be.



############ What it means to be clean/malware/reference ########

# The following two variables list the various names under which
# a file is known as clean/malware. More can be added in the future.
# The intersection of the two lists should be empty!
classificationNamesClean = ['clean']
classificationNamesMalware = ['malware']
classificationNamesReference = ['reference']



######### Querying and Inserting in Cassandra #########

# Which features to use to get a verdict and confidence
# From experiments, just feature 59 gives very good results,
# and seems to be highly correlated to many other features.
# If computational time allows, more features could be used.
# Time to get a verdict is proportional to the number of features used.
# Given as a list of strings!
confUsedFeatures = ['59']

# The number of hashes to use in the MinHash approximation
# This number is used for each of the features (i.e. it will in total
# check against len(featuresUsed)*refSamplesMaxNumberHashes).
# 2000 is probably too much (the gain in precision becomes negligible
# above a certain value), but do not go below 100.
confRefSamplesMaxNumberHashes = 1000

# The number of files from the reference set (which lies in Cassandra) to
# compare against. Ideally, this should equal to the whole set that is in
# Cassandra. In practice, depending on the Cassandra cluster speed at answering
# the requests, this number should be lowered to something giving reasonable 
# computational times.
confMaxRefSamplesCheckedAgainst = 10000

# Has to be smaller than the total number of samples we check against
# 1000 is a good value (probably too large) to make sure we have enough
# neighbours to have one of the opposite class among the lot
confNumberNeighborsReturned = min(10000, confMaxRefSamplesCheckedAgainst)


############### Neural Network (ELM) Parameters #################
confDictActivationFunctions = {'np.sin': np.sin, 'np.tanh': np.tanh}
confDictActivationFunctionsReverse = {np.sin: 'np.sin', np.tanh: 'np.tanh'}
confELMNumberNeurons = 500
confELMNumSamples = 5000
confELMActivationFunction = 'np.tanh'

confThresholdClassifier1=0.9999
confThresholdClassifier2=0
