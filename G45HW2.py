import sys
from pyspark import SparkContext, SparkConf
from pyspark.mllib.clustering import KMeans
import numpy as np

# giova
def MRComputeStandardObjective(U, C):
    """
This function computes the standard K-means clustering cost function on a set of points U with centers C.

Parameters
----------
U : pyspark.RDD
    The set of data points as (pos, category) where pos is a vector and category is either "A" or "B".
C : iterable
    The centers

Returns
-------
float
    The value of the cost function
    """

    d = U.map(lambda x : np.min([np.square(np.array(x[0])-c).sum() for c in C]))
    return d.mean()

def MRComputeFairObjective(U, C):
    """
This function computes the fair K-means clustering cost function on a set of points U with centers C.

Parameters
----------
U : pyspark.RDD
    The set of data points as (pos, category) where pos is a vector and category is either "A" or "B".
C : iterable
    The centers

Returns
-------
float
    The value of the fair cost function
    """
    DeltaA = MRComputeStandardObjective(U.filter(lambda x : x[1] == 'A'), C)
    DeltaB = MRComputeStandardObjective(U.filter(lambda x : x[1] == 'B'), C)
    return max(DeltaA, DeltaB)

def MRFairLloyd(U, K, M):
    """
This function computes the fair K-means clustering cost function on a set of points U with centers C.

Parameters
----------
U : pyspark.RDD
    The set of data points as (pos, category) where pos is a vector and category is either "A" or "B".
K : int
    The number of centers to be computed
M : int
    The number of iterations to be run of the algorithm

Returns
-------
list of the centers
    """
    C = [np.array(p) for (p,c) in U.take(K)]

    UA = U.filter(lambda x : x[1] == 'A')
    UB = U.filter(lambda x : x[1] == 'B')
    countA = UA.count()
    countB = UB.count()

    for i in range(M):
        T = 10; gamma = 0.5
        x = np.zeros(K)
        print("C:", C)
        partitionsA = UA.map(lambda x : (np.argmin([np.square(np.array(x[0])-c).sum() for c in C]),x[0])).groupByKey()
        a = np.array(partitionsA.map(lambda p : len(p)/countA).collect())
        muA = np.array(partitionsA.mapValues(lambda p : np.array(list(p)).mean(axis=0)).sortByKey().map(lambda x : x[1]).collect())
        # print("muA:", muA)

        partitionsB = UB.map(lambda x : (np.argmin([np.square(np.array(x[0])-c).sum() for c in C]),x[0])).groupByKey()
        b = np.array(partitionsB.map(lambda p : len(p)/countB).collect())
        muB = np.array(partitionsB.mapValues(lambda p : np.array(list(p)).mean(axis=0)).sortByKey().map(lambda x : x[1]).collect())
        # print("muB:", muB)
        l = np.linalg.norm(muA-muB,axis=1)
        # print("l:", l)
        for t in range(T):
            x = ((1-gamma)*b*l)/(gamma*a+(1-gamma)*b)
            FA = MRComputeStandardObjective(UA, x)
            FB = MRComputeStandardObjective(UB, x)
            gamma += (1 if FA > FB else -1)*(0.5)**(-t-2)
        C = ((l-x)[:,np.newaxis]*muA + x[:,np.newaxis]*muB)/l[:,np.newaxis]
    return C




def parse_line(line):
    parts = line.strip().split(',')
    point = tuple(float(x) for x in parts[:-1])
    group = parts[-1]
    return (point, group)

def main():
    if len(sys.argv) != 5:
        print("Usage: G45HW1.py <file_path> <L> <K> <M>")
        sys.exit(1)

    file_path, L, K, M = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])

    sc = SparkContext(appName="FairKMeans")

    points_rdd = sc.textFile(file_path, minPartitions=L).map(parse_line).cache()

    N = points_rdd.count()
    NA = points_rdd.filter(lambda x: x[1] == 'A').count()
    NB = points_rdd.filter(lambda x: x[1] == 'B').count()

    print(f"======= OUTPUT FOR L = {L}, K = {K}, M = {M} =======")
    print(f"\nInput file = {file_path}, L = {L}, K = {K}, M = {M}")
    print(f"N = {N}, NA = {NA}, NB = {NB}")

    vectors_rdd = points_rdd.map(lambda x: tuple(x[0]))
    model = KMeans.train(vectors_rdd, K, maxIterations=M)
    centroids = model.clusterCenters

    fair_centroids = MRFairLloyd(points_rdd, K, M)

    delta = MRComputeStandardObjective(points_rdd, centroids)
    phi = MRComputeFairObjective(points_rdd, centroids)

    print(f"Delta(U,C) = {delta:.6f}")
    print(f"Phi(A,B,C) = {phi:.6f}")

    # MRPrintStatistics(points_rdd, centroids)

    sc.stop()

if __name__ == "__main__":
    main()
