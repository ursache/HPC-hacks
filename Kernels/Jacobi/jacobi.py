import numpy  
from numpy import linalg as la
#
from pyspark import SparkContext
import time
#import matplotlib.pyplot as plt

# author gilles.fourestey@epfl.ch
# based on the code from http://www.dursi.ca/hpc-is-dying-and-mpi-is-killing-it/

def simulation(sc, nx, ny, nsteps, leftX = -1., rightX = +1., downY = -1., upY = +1., nu=.1):
    dx = (rightX - leftX)/(nx - 1)
    dy = (upY    - downY)/(ny - 1)
    mu = 0.25
    #
    def posFromIndex(i):
	iy    = i/nx
        ix    = i%nx
        x     = leftX + ix*dx
        y     = downY + iy*dy
	return [(x,y)]	
    #
    def interior(ip):                        # boilerplate
	#
	iy 	= ip[0]/nx
	ix	= ip[0]%nx
	return (ix > 0) and (ix < nx - 1) and (iy > 0) and (iy < ny - 1)
    	#
    def valFromIndex(i):
	#
	x = filter(lambda x: not interior(x), [(i, 1.)])
	y = filter(lambda x: 	 interior(x), [(i, 0.)])
	x.extend(y)
	#
        return x[0];
    #
    def stencil(item):
        i,t = item
	bvals = [(i, t)]
        vals = [ (i - 1, mu*t), (i + 1, mu*t), (i - ny, mu*t), (i + ny, mu*t) ]
        return filter(lambda x: not interior(x), bvals) + filter(interior, vals)
    #
    print "Stencil: ", nx, "x", ny, "\n"
    u_tmp = map(valFromIndex, range(nx*ny))
    un = sc.parallelize(u_tmp)#.partitionBy(nprocs)
    uo = zip(*(un.collect()));
    print "IC: "
    start = time.clock()
    for step in xrange(nsteps):
        stencilParts = un.flatMap(stencil)
        un     = stencilParts.reduceByKey(lambda x, y: x + y)
	collect = un.collect();
	myzip   = zip(*collect)
	diff=tuple(numpy.subtract(myzip[1] , uo))
	print step, ": l2 norm = ", la.norm(diff, ord=2)
	uo = myzip[1]
    end = time.clock()
    print "Final: "
    print "time = ", (end - start)

if __name__ == "__main__":  
    sc = SparkContext(appName="SparkDiffusion")
    sc.setLogLevel("OFF")
    ncores = sc.defaultParallelism
    print "\nHello friend...\n"
    print "number of cores = ", ncores 
    simulation(sc = sc, nx = 1000, ny = 1000, nsteps = 20)
