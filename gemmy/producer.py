"""
An abstract producer module running a job generator constantly. 

@author: Yoan Miche
@email: yoan.miche@aalto.fi
@version: 1.0
"""

# Global imports
import time

# Local imports
import queue

# @TODO : This generates a way too busy process when no files are to be watched anymore


class Producer:
    """This producer runs the job generator constantly.
       The list of jobs returned by the job generator is submitted to the given queue."""

    def __init__(self, queueName, queueHost, queuePort, jobGenerator, arguments):
        if not hasattr(jobGenerator, '__call__'): raise TypeError('Given job generator is not callable.')
        self.queueHost = queueHost
        self.queuePort = queuePort
        self.jobGenerator = jobGenerator
        self.arguments = arguments
        queueClass = queue.getQueueClass(queueName)
        self.queue = queueClass(queueHost, queuePort)


    
    def run(self):
        """ Producer running constantly the job_generator function
            and submitting the jobs to the requested queue."""
        while True:
            jobList = self.jobGenerator(*self.arguments)

            for job in jobList:
                # This simulates incoming samples, remove for production eventually
                time.sleep(0.01)
                if __debug__: print 'Submitting job %s to queue, with name %s, priority %s, body %s and sequence %s.' % (job, job.name, job.priority, job.body, job.sequence)

                self.queue.put(job)

