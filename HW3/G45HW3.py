from pyspark import SparkContext, SparkConf, StorageLevel
from pyspark.streaming import StreamingContext
import sys
import threading
import random

# Costanti per hash functions
p = 8191

# Hash function generatore
def make_hash_func(C):
    a = random.randint(1, p-1)
    b = random.randint(0, p-1)
    def h(x):
        return ((a * x + b) % p) % C
    return h

# Hash function per Count Sketch, con segno +/-1
def make_hash_func_sign(C):
    a = random.randint(1, p-1)
    b = random.randint(0, p-1)
    def h(x):
        idx = ((a * x + b) % p) % C
        # segno: restituisce 1 o -1 basato sul bit meno significativo di idx (ad es)
        sign = 1 if (idx % 2 == 0) else -1
        return idx, sign
    return h

def process_batch(time, batch):
    global streamLength, freq, CM, CS, h_CM, h_CS_h, h_CS_g, D, W, stopping_condition

    batch_size = batch.count()
    if streamLength[0] >= T:
        return
    streamLength[0] += batch_size

    # Raccogli gli elementi come int
    items = batch.map(lambda s: int(s)).collect()

    # Aggiorna frequenze esatte
    for x in items:
        freq[x] = freq.get(x, 0) + 1

        # Update Count-Min Sketch
        for i in range(D):
            idx = h_CM[i](x)
            CM[i][idx] += 1

        # Update Count Sketch
        for i in range(D):
            idx = h_CS_h[i](x)
            _, sign = h_CS_g[i](x)
            CS[i][idx] += sign

    if streamLength[0] >= T:
        stopping_condition.set()

if __name__ == "__main__":
    # Lettura parametri
    if len(sys.argv) != 6:
        print("USAGE: portExp T D W K")
        sys.exit(1)

    portExp = int(sys.argv[1])
    T = int(sys.argv[2])
    D = int(sys.argv[3])
    W = int(sys.argv[4])
    K = int(sys.argv[5])

    # Setup Spark
    conf = SparkConf().setMaster("local[*]").setAppName(f"GxxHW3_{portExp}")
    sc = SparkContext(conf=conf)
    ssc = StreamingContext(sc, 0.01)
    ssc.sparkContext.setLogLevel("ERROR")

    streamLength = [0]
    freq = {}

    # Inizializza strutture CM e CS
    CM = [[0]*W for _ in range(D)]
    CS = [[0]*W for _ in range(D)]

    # Crea hash functions
    h_CM = [make_hash_func(W) for _ in range(D)]
    h_CS_h = [make_hash_func(W) for _ in range(D)]
    h_CS_g = [make_hash_func_sign(W) for _ in range(D)]

    stopping_condition = threading.Event()

    # Crea stream
    stream = ssc.socketTextStream("algo.dei.unipd.it", portExp, StorageLevel.MEMORY_AND_DISK)
    stream.foreachRDD(lambda time, batch: process_batch(time, batch))

    print(f"Port = {portExp} T = {T} D = {D} W = {W} K = {K}")

    ssc.start()
    stopping_condition.wait()
    ssc.stop(False, False)


    # Calcolo risultati finali

    # Numero totale elementi e distinti
    total_items = streamLength[0]
    distinct_items = len(freq)

    # Trova top-K heavy hitters in freq
    sorted_freq = sorted(freq.items(), key=lambda x: x[1], reverse=True)
    if K > 0 and K <= distinct_items:
        kth_freq = sorted_freq[K-1][1]
    else:
        kth_freq = 0

    # Lista heavy hitters con freq >= kth_freq
    heavy_hitters = [item for item in sorted_freq if item[1] >= kth_freq]

    # Calcola errori medi relativi
    def estimate_CM(x):
        return min(CM[i][h_CM[i](x)] for i in range(D))
    def estimate_CS(x):
        estimates = []
        for i in range(D):
            idx = h_CS_h[i](x)
            _, sign = h_CS_g[i](x)
            estimates.append(CS[i][idx] * sign)
        # Mediana delle stime
        estimates.sort()
        mid = len(estimates) // 2
        if len(estimates) % 2 == 1:
            return estimates[mid]
        else:
            return (estimates[mid-1] + estimates[mid]) / 2

    errors_CM = []
    errors_CS = []

    for (x, fx) in heavy_hitters:
        f_CM = estimate_CM(x)
        f_CS = estimate_CS(x)
        err_CM = abs(fx - f_CM) / fx
        err_CS = abs(fx - f_CS) / fx
        errors_CM.append(err_CM)
        errors_CS.append(err_CS)

    avg_err_CM = sum(errors_CM) / len(errors_CM) if errors_CM else 0
    avg_err_CS = sum(errors_CS) / len(errors_CS) if errors_CS else 0

    # Output formattato nello stile richiesto
    print(f"Port = {portExp} T = {T} D = {D} W = {W} K = {K}")
    print(f"Number of processed items = {total_items}")
    print(f"Number of distinct items  = {distinct_items}")
    print(f"Number of Top-K Heavy Hitters = {K}")
    print(f"Avg Relative Error for Top-K Heavy Hitters with CM = {avg_err_CM:.15f}")
    print(f"Avg Relative Error for Top-K Heavy Hitters with CS = {avg_err_CS:.15E}")

    if K <= 10:
        print("Top-K Heavy Hitters:")
        for i in range(K):
            x, fx = sorted_freq[i]
            f_CM = estimate_CM(x)
            print(f"Item {x} True Frequency = {fx} Estimated Frequency with CM = {f_CM}")


