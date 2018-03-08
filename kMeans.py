#!/usr/bin/python3
from matplotlib import pyplot as plt
import numpy as np

def eucDis(a,b):
    return np.linalg.norm(a-b)

plt.ion()

class DynamicUpdate():
    #Suppose we know the x range
    min_x = 0
    max_x = 10
    def on_launch(self,numCluster):
        #Set up plot
        self.numCluster = numCluster
        self.figure, self.ax = plt.subplots()
        colors = ['r', 'g', 'k']
        self.lines = []
        self.cents = []
        for i in range(self.numCluster):
            self.lines.append(self.ax.plot([], [], colors[i] + 'o')[0])
            self.cents.append(self.ax.plot([], [], colors[i] + '^',markersize=15)[0])
        #Autoscale on unknown axis and known lims on the other
        self.ax.set_autoscaley_on(True)
        #self.ax.set_xlim(self.min_x, self.max_x)
        self.ax.grid()

    def on_running(self, centroids, pts, belongsTo):
        #Update data (with the new _and_ the old points)
        for ind, centr in enumerate(centroids):
            clusPts = pts[np.nonzero(belongsTo == ind)[0],:]
            self.lines[ind].set_data(clusPts[:,0], clusPts[:,1])
            self.cents[ind].set_data(centr[0], centr[1])
        #Need both of these in order to rescale
        self.ax.relim()
        self.ax.autoscale_view()
        #We need to draw *and* flush
        self.figure.canvas.draw()
        self.figure.canvas.flush_events()
        plt.pause(0.5)

def kMeans(numCluster, data, epsilon=0, distFucn=eucDis):
    numSamples, numDims = data.shape
    clusterAssign = np.zeros(numSamples)

    # Creating and Initializing the plots
    d = DynamicUpdate()
    d.on_launch(numCluster)

    # Old centroids
    oldCent = np.zeros((numCluster,numDims))

    # Initializing new centroids
    newCent = np.zeros((numCluster,numDims))
    while True:
        start = np.random.randint(numSamples, size=numDims)
        start = np.unique(start)
        if start.shape[0] == numCluster:
            break
        else:
            pass

    for i in range(numCluster):
        newCent[i,:] = data[start[i],:]


    # Iterating till convergence
    while eucDis(newCent, oldCent) > epsilon:
        for ptInd, pt in enumerate(data):
            ptDis = np.zeros(numCluster)
            for centInd, cent in enumerate(newCent):
                ptDis[centInd] = distFucn(newCent[centInd], data[ptInd])
            clusterAssign[ptInd] = np.argmin(ptDis)

        tempCent = np.zeros((numCluster, numDims))
        for j in range(numCluster):
            clusPts = data[np.nonzero(clusterAssign == j)[0],:]
            tempCent[j] = np.mean(clusPts, axis=0)

        oldCent, newCent = newCent, tempCent

        # Updaing the plots
        d.on_running(newCent, data, clusterAssign)
    return newCent


data = np.loadtxt('twoClus.txt')

print('Final centroids are : \n', kMeans(2, data))
