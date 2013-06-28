"""
Abstract consumer class.

@author: Yoan Miche
@email: yoan.miche@aalto.fi
@version: 1.0
"""

import time

import queue
from queue import Queue
from job import Job
import gemmy
import json


# This function is only for testing purposes: Creates a JSON based on a file on disk
# to simulate the incoming flow of jobs from the queue for later
def jobToJSONAndCassandra(job):
    """ Function to help testing. Not to be used for production.
        Simulates an incoming flow of data from SMA, in the form of JSON
        data. Basically reads a file from disk and creates a JSON structure 
        fed to the rest.

    """

    JSONData = {"priority":"", "sha1":"", "preprocessor_data": {}, "metadata":{}}

    JSONData["priority"] = 0
    JSONData["sha1"] = job.body.split('/')[-1].split('.')[0]
    JSONData["preprocessor_data"] = { "classification_status":"unknown",\
                                      "pp_results":{}\
                                    }
    JSONData["metadata"] = { "sha1":"",\
                             "system":"",\
                             "first_seen":"",\
                             "sha256":"",\
                             "md5":"",\
                             "size":0}

    fileName = job.body
    file = open(fileName, 'r')

    metadataFileName = fileName.split('.')[0]+'.metadata'
    metadataFile = open(metadataFileName, 'r')

    for line in metadataFile.readlines():
        key = line.split(' ')[0]
        values = line.split(' ')[1:]
        value = ''
        for item in values:
            value += item+' '
        value = value[:-2]
        if key == "first_seen":
            JSONData["metadata"]["first_seen"] = value
        if key == "md5":
            JSONData["metadata"]["md5"] = value
        if key == "sha1":
            JSONData["metadata"]["sha1"] = value
            JSONData["metadata"]["sha256"] = value
        if key == "size":
            JSONData["metadata"]["size"] = int(value)
        if key == "systems":
            JSONData["metadata"]["system"] = value
            JSONData["metadata"]["classification"] = value
    metadataFile.close()

    for line in file.readlines():
        featureNumber, featureValue = line.split(' ')
        if ',' in featureValue:
            featureValue = featureValue.split(',')[1]
        featureValue = featureValue.rsplit()[0]
        if not JSONData["preprocessor_data"]["pp_results"].has_key(str(featureNumber)):
            JSONData["preprocessor_data"]["pp_results"][str(featureNumber)] = []
        JSONData["preprocessor_data"]["pp_results"][str(featureNumber)].append(featureValue)

    file.close()
    task = gemmy.GemmyTask(JSONData)
    
    job.sequence.pop()
    return job, task


class Consumer:
    """ This consumer polls for tasks from a queue.
        Additional parameter prefTube gives a priority to
        a certain type of tasks, if available.
    """

    def __init__(self, queueName, queueHost, queuePort, prefTube='default'):
        self.queueHost = queueHost
        self.queuePort = queuePort
        queueClass = queue.getQueueClass(queueName)
        self.queue = queueClass(self.queueHost, self.queuePort)
        self.gemmy = gemmy.Gemmy()
    	
        if (prefTube == 'default') or (prefTube in self.queue.tubes()):
            self.prefTube = str(prefTube)
        else:
            if __debug__: print 'Unknown task type at the moment, reverting to default.'
            self.prefTube = 'default'
    

    def run(self):
        if __debug__: print 'Polling for tasks from queue.'
        while True:
            # If the preferred task is not default (i.e. not anything)
            if self.prefTube != 'default':
                # If the tube to watch for is not empty, work on it exclusively
                if self.queue.jobsInTube != 0:
                    job = self.queue.pull(self.prefTube)
                # Requested working tube is empty, do something on any other for this round
                else: 
                    tube = self.queue.getNonEmptyTube()
                    if tube != 'default':
                        job = self.queue.pull(tube)
                    else:
                        job = ''

            # Preferred task is default, work on any tube available
            else:
                tube = self.queue.getNonEmptyTube()
                print 'No prefered tube, using non empty one, if available: %s' % tube
                if tube != 'default':
                    job = self.queue.pull(tube)
                
            if tube != 'default':
                    if __debug__: print 'Processing task from queue %s with body: %s' % (job.sequence[-1], job.body)
                    #self.task_handler = gemmy.getTaskHandler(job.sequence[-1])
                    if __debug__: print 'Job sequence (reversed) is %s. Executing %s' % (job.sequence, 'Gemmy')
                    
                    job, task = jobToJSONAndCassandra(job)
                    SMAFact = self.gemmy.run(task)

                    # Added for tests
                    if not task.isReference():
                        neighborsFileName = job.body.split('.')[0]+'.neighbors'
                        neighborsFile = open(neighborsFileName, 'w')
                        for neighbor in self.gemmy.neighbors:
                            neighborsFile.write(str(neighbor[0])+' '+str(neighbor[1])+'\n')
                        neighborsFile.close()


                    print SMAFact
                    
                    if job.sequence != ['null']:
                        if job.sequence.__len__() != 0:
                            if __debug__: print 'Processing done; job submitted to next tube %s.' % job.sequence[-1]
                            self.queue.put(job)
            
            time.sleep(0.01) # Just to prevent busy loop in case of problems in loop.
   