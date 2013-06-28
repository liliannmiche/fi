"""
A Job class.
Defines the contents of a job for the queue.

@author: Yoan Miche
@email: yoan.miche@aalto.fi
@version: 1.0
"""

class Job:
    """ Defines the body of a job."""

    def __init__(self, name='null', priority=0, body='null', sequence=['null']):
        self.sequence = sequence
        self.name = name
        self.priority = priority
        self.body = body
        self.sequence.reverse()
        if __debug__: print 'Initializing job %s with sequence (reversed) %s' % (self.name, self.sequence)

    def statistics(self):
        """ Prints job details."""
        print 'Job %s with priority %s:' % (self.name, self.priority)
        print '\t Body %s' % self.body
        print '\t Work sequence (inverted): %s' % self.sequence


def stringToJob(jobToParse, delimiter=':'):
    """ Parses a string to a job.
        Returns a job with the parameters from the jobToParse.
    """
    if not jobToParse.split(delimiter).__len__() == 4: raise ValueError('Given string does not contain a complete/valid job.')
    name, priority, body, sequence = jobToParse.split(delimiter)
    priority = int(priority)
    sequence = eval(sequence)
    sequence.reverse()
    return Job(name=name, body=body, priority=priority, sequence=sequence)

def jobToString(job, delimiter=':'):
    """ Creates a string out of a job for the queue.
        Returns a string.
    """
    if not isinstance(job, Job): raise TypeError('Given job is not of valid type.')
    stringJob = str(job.name)+delimiter+str(job.priority)+delimiter+str(job.body)+delimiter+str(job.sequence)
    return stringJob