"""
Consumer Starter.

@author: Yoan Miche
@email: yoan.miche@aalto.fi
@version: 1.0
"""

import sample
import queue
import job
import consumer


def main():
    """ Called if this file is run from the command line directly."""

    host = '130.233.178.108'
    port = 14712

    bstkdq = queue.BeanstalkdQueue(host, port)
    bstkdq.statistics()

    queueHost = host
    queuePort = port
    queueName = 'bs'
    myconsumer = consumer.Consumer(queueName, queueHost, queuePort)
    myconsumer.run()



if __name__ == '__main__':
    main()
