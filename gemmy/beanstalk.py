"""
Simple Beanstalk client helper wrapper.

Provides the same API as beanstalkc library but with synchronization
and connection reset functionality when socket fails.

@author: Hannu Maaranen
@copyright: F-Secure Corporation
"""

from __future__ import with_statement
import beanstalkc
import sys
import threading
import traceback

def bs_wrapped(method):
    """Wraps a beanstalk call into synchronized and connection problem reseting decorator.
    Requires that the object has self.mutex and self.init_beanstalkc methods.
    """
    def bs_method(self, *args, **kw):
        try:
            with self.mutex:
                return method(self, *args, **kw)
        except beanstalkc.SocketError:
            traceback.print_exc( file=sys.stdout )
            print "Beanstalk connection died. Reseting connection."
            self.init_beanstalkc()
            with self.mutex:
                # Not catching the error the second time, as socket error should be fixed now.
                return method(self, *args, **kw)
    
    return bs_method

class SynchBeanstalkClient(object):
    """
    Wraps the default Beanstalk connection into helper class that reconnects
    and resets the connection in case problems occur.
    Synchronizes the Beanstalk usage, as by default beanstalkc client is not thread safe.
    """
    
    def __init__(self, bs_host="127.0.0.1", bs_port=14711):
        self.mutex = threading.Lock()
        self.bs_host = bs_host
        self.bs_port = bs_port
        self.bs = None
        
        self.__watched = set(["default"])
        self.__using = "default"
        
        self.init_beanstalkc()
    
    def init_beanstalkc(self):
        """Initializes and created new beanstalk connection.
        Call this if your connection dies whenever.
        """
        with self.mutex:
            if self.bs:
                self.bs.close()
            self.bs = beanstalkc.Connection(host=self.bs_host, port=self.bs_port)
            self.bs.use(self.__using)
            for x in self.__watched:
                self.bs.watch(x)
        self.watching() # Confirms the correct tubes are watched.
    
    @bs_wrapped
    def put(self, task_str, priority=50, ttr=120):
        """Puts a task into beanstalk queue."""
        return self.bs.put(task_str, priority=priority, ttr=ttr)
    
    @bs_wrapped
    def reserve(self, timeout=None):
        """Reserves a task from a beanstalk queue."""
        return self.bs.reserve(timeout=timeout)
    
    @bs_wrapped
    def close(self):
        """Closes the connection to Beanstalk server."""
        if self.bs:
            self.bs.close()
        self.bs = None
    
    @bs_wrapped
    def kick(self, bound=1):
        """Kick at most bound jobs into the ready queue."""
        return self.bs.kick(bound)
    
    @bs_wrapped
    def peek(self, jid):
        """Peek at a job. Returns a Job, or None."""
        return self.bs.peek(jid)

    @bs_wrapped
    def peek_ready(self):
        """Peek at next ready job. Returns a Job, or None."""
        return self.bs.peek_ready()
    
    @bs_wrapped
    def peek_delayed(self):
        """Peek at next delayed job. Returns a Job, or None."""
        return self.bs.peek_delayed()

    @bs_wrapped
    def peek_buried(self):
        """Peek at next buried job. Returns a Job, or None."""
        return self.bs.peek_buried()

    @bs_wrapped
    def tubes(self):
        """Return a list of all existing tubes."""
        return self.bs.tubes()

    @bs_wrapped
    def using(self):
        """Return a list of all tubes currently being used."""
        result = self.bs.using()
        assert self.__using == result, "Invalid using value: %s. Expected: %s" % (self.__using, result)
        return result

    @bs_wrapped
    def use(self, name):
        """Use a given tube."""
        self.__using = self.bs.use(name)
        return self.__using

    @bs_wrapped
    def watching(self):
        """Return a list of all tubes being watched."""
        result = self.bs.watching()
        assert list(self.__watched) == result, \
               "Watching miss-match. Has %s, expected: %s" % (list(self.__watched), result)
        return result

    @bs_wrapped
    def watch(self, name):
        """Watch a given tube."""
        result = self.bs.watch(name)
        if result > 0:
            self.__watched.add(name)
        return result

    @bs_wrapped
    def ignore(self, name):
        """Stop watching a given tube."""
        result = self.bs.ignore(name)
        if name in self.__watched:
            self.__watched.remove(name)
        return result

    @bs_wrapped
    def stats(self):
        """Return a dict of beanstalkd statistics."""
        return self.bs.stats()

    @bs_wrapped
    def stats_tube(self, name):
        """Return a dict of stats about a given tube."""
        return self.bs.stats_tube(name)

    @bs_wrapped
    def pause_tube(self, name, delay):
        """Pause a tube for a given delay time, in seconds."""
        self.bs.pause_tube(name, delay)

    @bs_wrapped
    def delete(self, jid):
        """Delete a job, by job id."""
        self.bs.delete(jid)

    @bs_wrapped
    def release(self, jid, priority=beanstalkc.DEFAULT_PRIORITY, delay=0):
        """Release a reserved job back into the ready queue."""
        self.bs.release(jid, priority, delay)

    @bs_wrapped
    def bury(self, jid, priority=beanstalkc.DEFAULT_PRIORITY):
        """Bury a job, by job id."""
        self.bs.bury(jid, priority)

    @bs_wrapped
    def touch(self, jid):
        """Touch a job, by job id, requesting more time to work on a reserved
        job before it expires."""
        self.bs.touch(jid)

    @bs_wrapped
    def stats_job(self, jid):
        """Return a dict of stats about a job, by job id."""
        return self.bs.stats_job(jid)

