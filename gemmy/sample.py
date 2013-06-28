"""
Sample class.
Currently not used. Was planned for a better way
to compute the Jaccard, but finally, Python is too slow.

Could be used to re-write properly some parts of the code,
especially in task_handlers and testing functions.

@author: Yoan Miche
@email: yoan.miche@aalto.fi
@version: 1.0
"""

import os
from os import path

class Sample:
    """ Defines the class that holds a sample's values.
        The sample's data is held as a dictionary of sets."""

    def __init__(self, filename):
        self.feature = {}
        self.filename = filename
        self.name = os.path.basename(filename).split(".")[0]
        f = open(filename, 'r')
        for line in f.readlines():
            feature_value, hash_value = line.split(" ")
            if self.feature.has_key(feature_value):
                if self.feature[feature_value].__len__() == 0:
                    self.feature[feature_value] = set()
            else:
                self.feature[feature_value] = set()
            
            self.feature[feature_value].add(hash_value)
        
        f.close()

    def getClass(self):
        """ Returns the class of the sample if the metadata file is available.
            Returns null otherwise.
        """
        filename_path = os.path.dirname(self.filename)

        try:
            f = open(os.path.join(filename_path, self.name+'.metadata'))
            for line in f.readlines():
                if 'clean' in line:
                    return 'clean'
            return 'malware'
        except Exception, err:
            return 'NULL'
            
    def statistics(self):
        """ Prints some statistics about the sample."""
        print "Sample %s has:" % self.name
        print "\t %s feature values," % self.feature.__len__()
        print "\t %s hash values" % sum([self.feature[i].__len__() for i in [j for j in self.feature]])



