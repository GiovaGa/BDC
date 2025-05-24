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

def gather_partitions(pts):
    """
    pts: iterable of tuples i,x
    where i int, x point
    """
    # print(list(pts))
    if len(list(pts)) == 0:
      return []

    dim = len(list(pts)[0][1])
    K = max([p[0] for p in pts]) + 1
    cnt = [0]*K
    ans = np.zeros((K,dim))
    for i,x in pts:
        cnt[i] += 1
        ans[i] += np.array(x)
    return [(i,(cnt[i], ans[i])) for i in range(K)]

def reduce_partitions(pts):
    """
    """
    # print(list(pts))
    if len(list(pts)) == 0:
      return []

    dim = len(list(pts)[0][1])
    cnt = int(0)
    ans = [0]*dim
    for s,x in pts:
        cnt += s
        ans += x
    return [(cnt,ans)]


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

    UA = U.filter(lambda x : x[1] == 'A').cache(); countA = UA.count()
    UB = U.filter(lambda x : x[1] == 'B').cache(); countB = UB.count()

    dim = len(C[0]) # number of dimensions of the points
    a, Ma = np.zeros(K), np.zeros((K,dim))
    b, Mb = np.zeros(K), np.zeros((K,dim))
    T = 10; gamma = 0.5

    for i in range(M):
        ret = UA.mapPartitions(lambda p : gather_partitions([(np.argmin([np.square(np.array(x[0])-c).sum() for c in C]), x[0]) for x in p])) \
                .groupByKey() \
                .mapValues(reduce_partitions) \
                .collect()
        for i,[(ai,mui)] in ret:
            a[i] = ai
            if ai > 0: Ma[i] = mui/ai
        a /= countA

        ret = UB.mapPartitions(lambda p : gather_partitions([(np.argmin([np.square(np.array(x[0])-c).sum() for c in C]),x[0]) for x in p])) \
                .groupByKey() \
                .mapValues(reduce_partitions) \
                .collect()
        for i,[(bi,mui)] in ret:
            b[i] = bi
            if bi > 0: Mb[i] = mui/bi
        b /= countB
        Ma[a == 0] = Mb[a == 0]
        Mb[b == 0] = Ma[b == 0]

        l = np.linalg.norm(Ma-Mb,axis=1)

        fixed_a = MRComputeStandardObjective(UA, Ma)/countA
        fixed_b = MRComputeStandardObjective(UB, Mb)/countB

        x = computeVectorX(fixed_a,fixed_b,a,b,l,K)

        C = [((l[i]-x[i])*Ma[i] + x[i]*Mb[i])/l[i] if l[i] > 0 else Ma[i] for i in range(K) ]
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

