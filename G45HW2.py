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

def gather_partitions(pts):
    """
    pts: iterable of tuples i,x
    where i int, x pair of int
    """
    # print(list(pts))
    K = max([p[0] for p in pts]) + 1
    cnt = np.zeros(K, np.int_)
    ans = np.zeros((K,2))
    for i,x in pts:
        cnt[i] += 1
        ans[i] += np.array(x[1])
    return [(i,(cnt[i], ans[i])) for i in range(K)]

def reduce_partitions(pts):
    """
    """

    # print(list(pts))
    cnt = int(0)
    ans = np.zeros(2)
    for s,x in pts:
        cnt += s
        ans += x
    return [(cnt,ans)]


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
    C = np.array([np.array(p) for (p,c) in U.takeSample(withReplacement=False,num=K,seed=69)])

    UA = U.filter(lambda x : x[1] == 'A'); countA = UA.count()
    UB = U.filter(lambda x : x[1] == 'B'); countB = UB.count()
    a, Ma = np.zeros(K), np.zeros((K,2))
    b, Mb = np.zeros(K), np.zeros((K,2))

    for i in range(M):
        T = 10; gamma = 0.5
        x = np.zeros(K)
        print("C:", C)
        ret = UA.mapPartitions(lambda p : gather_partitions([(np.argmin([np.square(np.array(x[0])-c).sum() for c in C]), x[0]) for x in p])) \
                .groupByKey() \
                .mapValues(reduce_partitions) \
                .collect()
        print(ret)
        for i,[(ai,mui)] in ret: a[i] = ai; Ma[i] = mui/ai
        a /= countA
        # print("a:", a)
        # print("Ma:", Ma)

        ret = UB.mapPartitions(lambda p : gather_partitions([(np.argmin([np.square(np.array(x[0])-c).sum() for c in C]),x[0]) for x in p])) \
                .groupByKey() \
                .mapValues(reduce_partitions) \
                .collect()
        for i,[(bi,mui)] in ret: b[i] = bi; Mb[i] = mui/bi
        b /= countB
        print("b:", b)
        print("Mb:", Mb)
        l = np.linalg.norm(Ma-Mb,axis=1)
        # print("l:", l)

        DeltaA = MRComputeStandardObjective(UA, x)/countA
        DeltaB = MRComputeStandardObjective(UB, x)/countB
        for t in range(T):
            x = ((1-gamma)*b*l)/(gamma*a+(1-gamma)*b)
            FA = DeltaA + np.dot(a,np.square(x))
            FB = DeltaB +np.dot(b,np.square(l-x))
            gamma += (1 if FA > FB else -1)*(0.5)**(-t-2)
        C = ((l-x)[:,np.newaxis]*Ma + x[:,np.newaxis]*Mb)/l[:,np.newaxis]
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
