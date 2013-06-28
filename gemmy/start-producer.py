"""
Producer Starter.

@author: Yoan Miche
@email: yoan.miche@aalto.fi
@version: 1.0
"""

import producer
from job_generator import gcFileWatcher


def main():
    """ Called if this file is run from the command line directly.
    """

    queueName = 'bs'
    queueHost = '130.233.178.108'
    queuePort = 14712

    jobGenerator = gcFileWatcher
    jobGeneratorParameters = ['/share/work/ymiche/F-Secure/jozsef-test/', ['mhcd']]

    myproducer = producer.Producer(queueName, queueHost, queuePort, jobGenerator, jobGeneratorParameters)
    myproducer.run()


if __name__ == '__main__':
    main()
