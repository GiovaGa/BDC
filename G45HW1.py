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

# simpatine
def point_count(list):

    clusters_dict = {}

    for c_id, category in list[1]:
        if c_id not in clusters_dict: clusters_dict[c_id] = np.array([0,0])
        if category == 'A':
          clusters_dict[c_id][0] += 1
        if category == 'B':
          clusters_dict[c_id][1] += 1

    return clusters_dict.items()

def MRPrintStatistics(U, C):
    """
This function prints the number of points that end up in each cluster, divided by category.

Parameters
----------
U : pyspark.RDD
    The set of data points as (pos, category) where pos is a vector and category is either "A" or "B".
C : iterable
    The centers

Prints
------
Triplets (c_i, NA_i, NB_i): respectively the i-th centroid in C, the number of points of category A in cluster i, and the number of points of category B in cluster i.
    """
    N = U.count()

    L = int(np.sqrt(N))

    triplets = (U.map(lambda x: (np.random.randint(0, L-1), (np.argmin([np.square(np.linalg.norm(np.array(x[0])-c)) for c in C]), x[1])))
                .groupByKey()
                .flatMap(point_count)
                .reduceByKey(lambda x, y: x + y))

    triplets_list = sorted(triplets.collect(), key=lambda x: x[0])

    for c_id, N_vec in triplets_list:

      print(f"i = {c_id}, center = (", end = "")

      print("%.6f" % C[c_id][0], end = "")

      for i in range(1, len(C[c_id])):
          print(",%.6f" % C[c_id][i], end = "")

      print(f"), NA{c_id} = {N_vec[0]}, NB{c_id} = {N_vec[1]}")

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

    '''centroids1 = [ # Centroids for example output 1
           np.array([40.750721,-73.980436]),
           np.array([40.724214,-74.193689]),
           np.array([40.779300,-73.428700]),
           np.array([40.663394,-73.786612]), ]
    centroids = [ # Centroids for example output 2
           np.array([40.749035,-73.984431]),
           np.array([40.873440,-74.192170]),
           np.array([40.693363,-74.178147]),
           np.array([40.746095,-73.830627]), ]'''

    delta = MRComputeStandardObjective(points_rdd, centroids)
    phi = MRComputeFairObjective(points_rdd, centroids)

    print(f"Delta(U,C) = {delta:.6f}")
    print(f"Phi(A,B,C) = {phi:.6f}")

    MRPrintStatistics(points_rdd, centroids)

    sc.stop()

if __name__ == "__main__":
    main()
