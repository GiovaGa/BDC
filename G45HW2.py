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
    for i in range(M):
        pass
    pass

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

    delta = MRComputeStandardObjective(points_rdd, centroids)
    phi = MRComputeFairObjective(points_rdd, centroids)

    print(f"Delta(U,C) = {delta:.6f}")
    print(f"Phi(A,B,C) = {phi:.6f}")

    MRPrintStatistics(points_rdd, centroids)

    sc.stop()

if __name__ == "__main__":
    main()
