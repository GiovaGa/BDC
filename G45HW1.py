import sys
from pyspark import SparkContext, SparkConf
from pyspark.mllib.clustering import KMeans
import numpy as np

# giova
def MRComputeStandardObjective(U, C):
    pass

def MRComputeFairObjective(U, C):
    pass

# simpatine
def MRPrintStatistics(U, C):
    pass

def parse_line(line):
    parts = line.strip().split(',')
    point = np.array([float(x) for x in parts[:-1]])  
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
    

    vectors_rdd = points_rdd.map(lambda x: np.array(x[0]))
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
