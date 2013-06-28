"""
A set of job generators for the producer.
Currently only a file watcher polling regularly for new files
with extension '.gc'.

@author: Yoan Miche
@email: yoan.miche@aalto.fi
@version: 0.1
"""

# Global imports
import os

# Local imports
from job import Job



def gcFileWatcher(dir, sequence):
    """ Polls recursively the directory given and checks for ".gc" files.
        Renames the gc files with a '.visited' extension.
        Returns a list of jobs. 
    """

    jobList = []

    if __debug__: print 'Watching jobs in %s to submit with sequence %s.' % (dir, sequence)

    for root, dirs, files in os.walk(dir):
        for file in files:
            if file.endswith('gc'):
                try:
                    os.rename(os.path.join(root, file), os.path.join(root, file)+'.visited')
                    workSequence = list(sequence)
                    job = Job(name=os.path.join(root, file), priority=0, body=os.path.join(root, file)+'.visited', sequence=workSequence)
                    jobList.append(job)
                    if __debug__: print 'Created job %s with name %s, priority %s, body %s and sequence %s.' % (job, job.name, job.priority, job.body, job.sequence)

                except Exception, ex:
                    print "Failed to rename found file %r for job creation. Exception: %s" % (os.path.join(root, file), ex)
                    continue

    return jobList


testFilesExtensions = ['gc']
def testFileWatcher(dir, sequence):
    """ Similar to the above file watcher, but for test files.
         Might consider some code factorization at some point.
    """
    jobList = []

    if __debug__: print 'Watching test jobs in %s to submit with sequence %s.' % (dir, sequence)

    for root, dirs, files in os.walk(dir):
        for file in files:
            if file.endswith(tuple(testFilesExtensions)):
                try:
                    if not file.endswith('visited'):
                        os.rename(os.path.join(root, file), os.path.join(root, file)+'.visited')
                    workSequence = list(sequence)
                    job = Job(name=os.path.join(root, file), priority=0, body=os.path.join(root, file)+'.visited', sequence=workSequence)
                    jobList.append(job)
                    if __debug__: print 'Created job %s with name %s, priority %s, body %s and sequence %s.' % (job, job.name, job.priority, job.body, job.sequence)

                except Exception, ex:
                    print "Failed to rename found file %r for job creation. Exception: %s" % (os.path.join(root, file), ex)
                    continue

    return jobList