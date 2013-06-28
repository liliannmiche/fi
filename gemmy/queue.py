"""
Queue management system.
Abstraction layer for Queue Management System.
Contains instantiations for some Queue Management Systems:
    - Beanstalkd
    - raise NotImplementedError :)

@author: Yoan Miche
@email: yoan.miche@aalto.fi
@version: 1.0
"""

# Global imports
import abc

# Local imports
# For the Beanstalkd implementation
from beanstalk import SynchBeanstalkClient
from job import Job, stringToJob, jobToString


class Queue:
    """ Class abstracting the queue operations.
        Instantiation gives it the specifics of the queue implementation.
        Added some useful for debugging queue information display methods.
    """
    __metaclass__ = abc.ABCMeta

    def __init__(self, host='127.0.0.1', port=8888):
        """ Global initialization of the class."""
        self.connect(host, port)
        print 'Connected to the QMS.'

    @abc.abstractmethod
    def connect(self, host, port):
        """ Connects to host on port for working with the queue."""
        raise NotImplementedError

    @abc.abstractmethod
    def put(self, job):
        """ Puts a job in the queue, in specific tube."""
        raise NotImplementedError
    
    @abc.abstractmethod
    def pull(self, tube):
        """ Pull a job from the queue, from specific tube.
            Returns a job."""
        raise NotImplementedError
    
    @abc.abstractmethod
    def tubes(self):
        """ Returns a list of tubes, unsorted."""
        raise NotImplementedError
    
    @abc.abstractmethod
    def tubeStatistics(self, tube):
        """ Prints specific tube statistics."""
        raise NotImplementedError

    @abc.abstractmethod
    def statistics(self):
        """ Prints Queue statistics, all tubes."""
        raise NotImplementedError

    @abc.abstractmethod
    def jobsInTube(self, tube):
        """ Returns how many jobs are waiting in given tube.
            If tube does not exist (yet or ever), return 0.
        """
        raise NotImplementedError
    
    @abc.abstractmethod
    def getNonEmptyTube(self):
        """ Returns the name of the first non-empty tube.
            If all are empty, returns default.
        """
        raise NotImplementedError
    
    
# Instantiations of the abstract Queue class


# @TODO: Check the pull function for timeout issues.
class BeanstalkdQueue(Queue):
    """ Instatiation of the Queue as a Beanstalkd one.
         Uses Hannu Maaranen's former wrapper code.
    """

    def connect(self, host, port):
        self.connection = SynchBeanstalkClient(host, port)

    def put(self, job):
        if not isinstance(job, Job): raise TypeError('Given job is not of valid type.')
        self.connection.use(job.sequence[-1])
        self.connection.put(jobToString(job), priority=job.priority)
        if __debug__: print 'Job %s put in tube %s.' % (job.name, job.sequence[-1]) 

    def pull(self, tube):
        for ignoredTube in self.tubes():
            self.connection.ignore(ignoredTube)
        self.connection.watch(tube)
        jobToParse = self.connection.reserve(timeout=10.0)
        # This is not safe, if the timeout is not enough, we call delete on an empty object
        jobToParse.delete()
        if __debug__: print jobToParse.body
        return stringToJob(jobToParse.body)

    def tubes(self):
        return self.connection.tubes()

    def tubeStatistics(self, tube):
        return self.connection.stats_tube(tube)

    def statistics(self):
        for tube in self.tubes():
            self.tubeStatistics(tube)
    
    def jobsInTube(self, tube):
        if not tube in self.tubes():
            if __debug__: print 'Given tube does not exist.'
            return 0
        else:
            return self.tubeStatistics(tube)['current-jobs-ready']
        
    def getNonEmptyTube(self):
        for tube in self.tubes():
            if self.jobsInTube(tube) != 0:
                return tube
        return 'default'



queues = { BeanstalkdQueue: ['bs', 'beanstalk', 'beanstalkd']}



# Helper function
def findKey(dictionary, value):
    """ Returns the key of dictionary given a value.
        Only gets the first key ! 
    """
    return [key for key, values in dictionary.iteritems() if value in values][0]



def getQueueClass(queue_name):
    """ Returns the right class name based on the queue name given.
        Raises exception if no class name is close enough.
    """
    if queue_name in [i for s in queues.values() for i in s]:
        return findKey(queues, queue_name)
    else:
        raise ValueError('Given queue name does not correspond to an existing implementation.')

