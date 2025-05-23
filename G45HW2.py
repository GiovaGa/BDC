import sys
import time
from pyspark import SparkContext, SparkConf
from pyspark.mllib.clustering import KMeans
import numpy as np
from computeVectorX import computeVectorX


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

    return U.map(lambda x : min([sum([(xi-ci)**2 for xi,ci in zip(x[0],c)]) for c in C])).mean()


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


def get_gather_partitions(K,dim,C):
    def gather_partitions(pts):
        """
        pts: iterable of tuples i,x
        where i int, x point
        """
        cnt = np.zeros(K)
        ans = np.zeros((K,dim))
        for x,_ in pts:
            i = np.argmin([np.square(np.array(x)-c).sum() for c in C])
            cnt[i] += 1
            ans[i] += np.array(x)
        return [(i,(cnt[i], ans[i])) for i in range(K)]
    return gather_partitions

def get_reduce_partitions(K,dim):
    def reduce_partitions(pts):
        """
        """
        cnt = int(0)
        ans = np.zeros(dim)
        for s,x in pts:
            cnt += s
            ans += x
        if cnt > 0: return [(cnt,ans/cnt)]
        else: return [(cnt,ans)]
    return reduce_partitions


def MRFairLloyd(U, K, M):
    """
    Implements the Fair K-Means Clustering algorithm.

    Parameters
    ----------
    U : pyspark.RDD
        The set of data points as (pos, category) where pos is a vector and category is either "A" or "B".
    k : int
        Number of clusters
    m : int
        Number of iterations

    Returns
    -------
    list
        Final set of centroids
    """
    vectors_rdd = U.map(lambda x: x[0])
    model = KMeans.train(vectors_rdd, K, maxIterations=0)
    C = model.clusterCenters

    L = U.getNumPartitions()
    UA = U.filter(lambda x : x[1] == 'A').repartition(L).cache(); countA = UA.count()
    UB = U.filter(lambda x : x[1] == 'B').repartition(L).cache(); countB = UB.count()

    dim = len(C[0]) # number of dimensions of the points
    a, Ma = np.zeros(K), np.zeros((K,dim))
    b, Mb = np.zeros(K), np.zeros((K,dim))
    T = 10; gamma = 0.5

    for i in range(M):
        ret = UA.mapPartitions(get_gather_partitions(K,dim,C)) \
                .groupByKey() \
                .mapValues(get_reduce_partitions(K,dim)) \
                .collect()
        for i,[(ai,mui)] in ret:
            a[i] = ai
            Ma[i] = mui
        a /= countA

        ret = UB.mapPartitions(get_gather_partitions(K,dim,C)) \
                .groupByKey() \
                .mapValues(get_reduce_partitions(K,dim)) \
                .collect()
        for i,[(bi,mui)] in ret:
            b[i] = bi
            Mb[i] = mui
        b /= countB
        Ma[a == 0] = Mb[a == 0]
        Mb[b == 0] = Ma[b == 0]

        l = np.linalg.norm(Ma-Mb,axis=1)

        fixed_a = MRComputeStandardObjective(UA, Ma)/countA
        fixed_b = MRComputeStandardObjective(UB, Mb)/countB

        x = computeVectorX(fixed_a,fixed_b,a,b,l,K)

        ids = l >= 0
        C = ((l[ids,np.newaxis]-x[ids,np.newaxis])*Ma[ids] + x[ids,np.newaxis]*Mb[ids])/l[ids,np.newaxis]
        C[~ids] = Ma[~ids]
    return C


def parse_line(line):
    parts = line.strip().split(',')
    point = tuple(float(x) for x in parts[:-1])
    group = parts[-1]
    return (point, group)


def main():
    if len(sys.argv) != 5:
        print("Usage: G45HW2.py <file_path> <L> <K> <M>")
        sys.exit(1)

    file_path, L, K, M = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])

    # Initialize Spark context
    conf = SparkConf().setAppName("FairKMeans")
    sc = SparkContext(conf=conf)

    # Print command-line arguments
    print(f"Input file = {file_path}, L = {L}, K = {K}, M = {M}")

    # Read input points into an RDD with L partitions
    points_rdd = sc.textFile(file_path, minPartitions=L).map(parse_line).cache()

    # Print points statistics
    N = points_rdd.count()
    NA = points_rdd.filter(lambda x: x[1] == 'A').count()
    NB = points_rdd.filter(lambda x: x[1] == 'B').count()
    print(f"N = {N}, NA = {NA}, NB = {NB}")

    # Compute standard Lloyd's centroids
    start_time = time.time()
    vectors_rdd = points_rdd.map(lambda x: x[0])
    model = KMeans.train(vectors_rdd, K, maxIterations=M)
    standard_centroids = model.clusterCenters
    standard_time = int((time.time() - start_time) * 1000)  # Convert to milliseconds

    # Compute fair Lloyd's centroids
    start_time = time.time()
    fair_centroids = MRFairLloyd(points_rdd, K, M)
    fair_time = int((time.time() - start_time) * 1000)  # Convert to milliseconds

    # Compute objective functions
    start_time = time.time()
    standard_obj = MRComputeFairObjective(points_rdd, standard_centroids)
    standard_obj_time = int((time.time() - start_time) * 1000)  # Convert to milliseconds

    start_time = time.time()
    fair_obj = MRComputeFairObjective(points_rdd, fair_centroids)
    fair_obj_time = int((time.time() - start_time) * 1000)  # Convert to milliseconds

    # Print results in the exact format required
    print(f"Fair Objective with Standard Centers = {standard_obj:.4f}")
    print(f"Fair Objective with Fair Centers = {fair_obj:.4f}")
    print(f"Time to compute standard centers = {standard_time} ms")
    print(f"Time to compute fair centers = {fair_time} ms")
    print(f"Time to compute objective with standard centers = {standard_obj_time} ms")
    print(f"Time to compute objective with fair centers = {fair_obj_time} ms")

    sc.stop()


if __name__ == "__main__":
    main()

