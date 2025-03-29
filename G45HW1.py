from pyspark import SparkContext, SparkConf

# giova
def MRComputeStandardObjective(U,C):
    pass

def MRComputeFairObjective(U,C):
    pass

# simpatine
def MRPrintStatistics(U,C):
    pass

# Giordano
def main():
    if len(sys.argv) != 5:
        print("Usage: G45HW1.py <input_path> <L> <K> <M>")
        sys.exit(1)
    
    input_path, L, K, M = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
    print(f"Arguments: input_path={input_path}, L={L}, K={K}, M={M}")
    
    sc = SparkContext(appName="G45HW1")
    input_rdd = sc.textFile(input_path, minPartitions=L).map(parse_point)
    
    N = input_rdd.count()
    NA = input_rdd.filter(lambda p: p[1] == 'A').count()
    NB = input_rdd.filter(lambda p: p[1] == 'B').count()
    print(f"N={N}, NA={NA}, NB={NB}")
    
    vectors_rdd = input_rdd.map(lambda p: p[0])
    kmeans_model = KMeans.train(vectors_rdd, K, maxIterations=M, initializationMode="k-means||")
    centroids = kmeans_model.clusterCenters
    
    delta = MRComputeStandardObjective(input_rdd, centroids)
    phi = MRComputeFairObjective(input_rdd, centroids)
    print(f"Delta(U,C)={delta}")
    print(f"Phi(A,B,C)={phi}")
    
    MRPrintStatistics(input_rdd, centroids)
    sc.stop()

if __name__ == "__main__":
    main()
