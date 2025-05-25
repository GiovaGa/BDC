from pyspark import SparkContext, SparkConf
from pyspark.streaming import StreamingContext
import sys
import random
import threading


P = 8191  

def generate_hash_function(C):
    a = random.randint(1, P - 1)  # a ∈ [1, p-1]
    b = random.randint(0, P - 1)  # b ∈ [0, p-1]
    def h(x):
        return ((a * x + b) % P) % C
    return h


def generate_count_sketch_hashes(D, W):
    h_funcs = [generate_hash_function(W) for _ in range(D)]
    g_funcs = []
    for _ in range(D):
        a = random.randint(1, P - 1)
        b = random.randint(0, P - 1)
        def g_factory(a, b):
            return lambda x: 1 if ((a * x + b) % P) % 2 == 0 else -1
        g_funcs.append(g_factory(a, b))
    return h_funcs, g_funcs

def update_count_min_sketch(CM, D, x, h_funcs):
    
    vals = [CM[i][h_funcs[i](x)] for i in range(D)]
    min_val = min(vals)
   
    for i in range(D):
        if vals[i] == min_val:
            CM[i][h_funcs[i](x)] += 1


def update_count_sketch(CS, D, x, h_funcs, g_funcs):
    for i in range(D):
        pos = h_funcs[i](x)
        sign = g_funcs[i](x)
        CS[i][pos] += sign


def estimate_cm_frequency(CM, D, x, h_funcs):
    return min(CM[i][h_funcs[i](x)] for i in range(D))


def estimate_cs_frequency(CS, D, x, h_funcs, g_funcs):
    estimates = []
    for i in range(D):
        pos = h_funcs[i](x)
        sign = g_funcs[i](x)
        estimates.append(CS[i][pos] * sign)
    estimates.sort()
    mid = len(estimates) // 2
    if len(estimates) % 2 == 0:
        return (estimates[mid-1] + estimates[mid]) / 2
    else:
        return estimates[mid]

def main():
    if len(sys.argv) != 6:
        print("USAGE: G45HW3.py portExp T D W K")
        sys.exit(1)

    portExp = int(sys.argv[1])
    T = int(sys.argv[2])
    D = int(sys.argv[3])
    W = int(sys.argv[4])
    K = int(sys.argv[5])

    
    conf = SparkConf().setMaster("local[*]").setAppName("G45HW3")
    sc = SparkContext(conf=conf)
    ssc = StreamingContext(sc, 0.01)
    ssc.sparkContext.setLogLevel("ERROR")

   
    global streamLength
    global histogram
    global CM
    global CS
    global h_cm
    global h_cs
    global g_cs

    streamLength = [0]  
    histogram = {}      

   
    CM = [ [0]*W for _ in range(D) ]
    CS = [ [0]*W for _ in range(D) ]

   
    h_cm = [generate_hash_function(W) for _ in range(D)]
    h_cs, g_cs = generate_count_sketch_hashes(D, W)

    stopping_condition = threading.Event()

    def process_batch(time, batch):
        if streamLength[0] >= T:
            return

        items = batch.collect()
        batch_size = len(items)

        if batch_size == 0:
            return

       
        for s in items:
            x = int(s)
            histogram[x] = histogram.get(x, 0) + 1

           
            update_count_min_sketch(CM, D, x, h_cm)

            
            update_count_sketch(CS, D, x, h_cs, g_cs)

        streamLength[0] += batch_size

        #print(f"Batch time {time}, size {batch_size}, total processed {streamLength[0]}")

        if streamLength[0] >= T:
            stopping_condition.set()

    
    stream = ssc.socketTextStream("algo.dei.unipd.it", portExp)

    stream.foreachRDD(process_batch)

    ssc.start()

    stopping_condition.wait()

    ssc.stop(stopSparkContext=True, stopGraceFully=True)

    num_distinct = len(histogram)

    sorted_freqs = sorted(histogram.values(), reverse=True)
    if len(sorted_freqs) >= K:
        phi_K = sorted_freqs[K-1]
    else:
        phi_K = 0

    heavy_hitters = [x for x,f in histogram.items() if f >= phi_K]

    def avg_relative_error(sketch_estimator):
        total_error = 0.0
        count = 0
        for x in heavy_hitters:
            true_f = histogram[x]
            est_f = sketch_estimator(x)
            rel_error = abs(true_f - est_f) / true_f if true_f != 0 else 0
            total_error += rel_error
            count += 1
        return total_error / count if count > 0 else 0.0

    avg_err_cm = avg_relative_error(lambda x: estimate_cm_frequency(CM, D, x, h_cm))
    avg_err_cs = avg_relative_error(lambda x: estimate_cs_frequency(CS, D, x, h_cs, g_cs))

    print("Input parameters:")
    print(f"Port: {portExp}, T: {T}, D: {D}, W: {W}, K: {K}")
    print(f"Number of distinct items: {num_distinct}")
    print(f"Average relative error CM on top-K heavy hitters: {avg_err_cm:.6f}")
    print(f"Average relative error CS on top-K heavy hitters: {avg_err_cs:.6f}")

    if K <= 10:

        top_k_sorted = sorted(heavy_hitters, key=lambda x: histogram[x], reverse=True)[:K]
        print("\nTop-K heavy hitters (true freq, estimated freq CM):")
        for x in top_k_sorted:
            true_f = histogram[x]
            est_f = estimate_cm_frequency(CM, D, x, h_cm)
            print(f"Item {x}: true freq = {true_f}, est freq CM = {est_f}")

if __name__ == "__main__":
    main()
