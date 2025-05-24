package com.example.g45hw2;

import java.util.*;
import java.util.stream.Collectors;
import java.io.Serializable;

import org.apache.spark.api.java.*;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.function.*;
import org.apache.spark.mllib.clustering.KMeans;
import org.apache.spark.mllib.clustering.KMeansModel;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.linalg.Vectors;
import scala.Tuple2;

public class G45HW2 {

    public static double[] computeVectorX(double fixedA, double fixedB, double[] alpha, double[] beta, double[] ell, int K) {
        double gamma = 0.5;
        double[] xDist = new double[K];
        double fA, fB;
        double power = 0.5;
        int T = 10;
        for (int t=1; t<=T; t++){
            fA = fixedA;
            fB = fixedB;
            power = power/2;
            for (int i=0; i<K; i++) {
                double temp = (1-gamma)*beta[i]*ell[i]/(gamma*alpha[i]+(1-gamma)*beta[i]);
                xDist[i]=temp;
                fA += alpha[i]*temp*temp;
                temp=(ell[i]-temp);
                fB += beta[i]*temp*temp;
            }
            if (fA == fB) {break;}
            gamma = (fA > fB) ? gamma+power : gamma-power;
        }
        return xDist;
    }

    public static class PointWithGroup implements Serializable {
        public Vector point;
        public String group;

        public PointWithGroup(Vector point, String group) {
            this.point = point;
            this.group = group;
        }
    }

    public static double computeStandardObjective(JavaRDD<PointWithGroup> U, Vector[] centers) {
        JavaRDD<Double> distances = U.map(p -> {
            double minDist = Double.MAX_VALUE;
            for (Vector c : centers) {
                double dist = squaredDistance(p.point, c);
                if (dist < minDist) minDist = dist;
            }
            return minDist;
        });
        return distances.reduce(Double::sum) / U.count();
    }

    public static double computeFairObjective(JavaRDD<PointWithGroup> U, Vector[] centers) {
        double deltaA = computeStandardObjective(U.filter(p -> p.group.equals("A")), centers);
        double deltaB = computeStandardObjective(U.filter(p -> p.group.equals("B")), centers);
        return Math.max(deltaA, deltaB);
    }

    public static Vector[] MRFairLloyd(JavaRDD<PointWithGroup> U, int K, int M) {
        JavaRDD<Vector> vectors = U.map(p -> p.point);
        KMeansModel model = KMeans.train(vectors.rdd(), K, 0); // Initial centers
        Vector[] C = model.clusterCenters();

        JavaRDD<PointWithGroup> UA = U.filter(p -> p.group.equals("A"));
        JavaRDD<PointWithGroup> UB = U.filter(p -> p.group.equals("B"));

        long countA = UA.count();
        long countB = UB.count();
        int dim = C[0].size();

        double[] a = new double[K];
        double[][] Ma = new double[K][dim];
        double[] b = new double[K];
        double[][] Mb = new double[K][dim];
        double[] l = new double[K];

        for (int iter = 0; iter < M; iter++) {
            Map<Integer, Tuple2<Double, double[]>> groupA = gatherAndReduce(UA, C, dim);
            for (int i = 0; i < K; i++) {
                Tuple2<Double, double[]> data = groupA.getOrDefault(i, new Tuple2<>(0.0, new double[dim]));
                a[i] = data._1();
                Ma[i] = scaleVector(data._2(), 1.0 / Math.max(a[i], 1e-9));
            }
            a = normalize(a, countA);

            Map<Integer, Tuple2<Double, double[]>> groupB = gatherAndReduce(UB, C, dim);
            for (int i = 0; i < K; i++) {
                Tuple2<Double, double[]> data = groupB.getOrDefault(i, new Tuple2<>(0.0, new double[dim]));
                b[i] = data._1();
                Mb[i] = scaleVector(data._2(), 1.0 / Math.max(b[i], 1e-9));
            }
            b = normalize(b, countB);

            for (int i = 0; i < K; i++) {
                if (a[i] == 0) Ma[i] = Mb[i];
                if (b[i] == 0) Mb[i] = Ma[i];
                l[i] = norm(diff(Ma[i], Mb[i]));
            }

            double fixedA = computeStandardObjective(UA, toVectors(Ma)) / countA;
            double fixedB = computeStandardObjective(UB, toVectors(Mb)) / countB;

            double[] x = computeVectorX(fixedA, fixedB, a, b, l, K);

            for (int i = 0; i < K; i++) {
                C[i] = Vectors.dense(
                        l[i] > 0
                        ? blend(Ma[i], Mb[i], x[i], l[i])
                        : Ma[i]
                );
            }
        }

        return C;
    }

    public static void main(String[] args) {
        if (args.length != 4) {
            System.err.println("Usage: FairKMeans <file_path> <L> <K> <M>");
            System.exit(1);
        }

        String filePath = args[0];
        int L = Integer.parseInt(args[1]);
        int K = Integer.parseInt(args[2]);
        int M = Integer.parseInt(args[3]);

        SparkConf conf = new SparkConf().setAppName("FairKMeans");
        JavaSparkContext sc = new JavaSparkContext(conf);

        JavaRDD<String> lines = sc.textFile(filePath, L);
        JavaRDD<PointWithGroup> data = lines.map(line -> {
            String[] parts = line.split(",");
            double[] coords = Arrays.stream(parts, 0, parts.length - 1).mapToDouble(Double::parseDouble).toArray();
            String group = parts[parts.length - 1];
            return new PointWithGroup(Vectors.dense(coords), group);
        }).cache();

        long N = data.count();
        long NA = data.filter(p -> p.group.equals("A")).count();
        long NB = data.filter(p -> p.group.equals("B")).count();
        System.out.printf("N = %d, NA = %d, NB = %d\n", N, NA, NB);

        long t0 = System.currentTimeMillis();
        KMeansModel standardModel = KMeans.train(data.map(p -> p.point).rdd(), K, M);
        Vector[] standardCenters = standardModel.clusterCenters();
        long t1 = System.currentTimeMillis();

        Vector[] fairCenters = MRFairLloyd(data, K, M);
        long t2 = System.currentTimeMillis();

        double stdObj = computeFairObjective(data, standardCenters);
        long t3 = System.currentTimeMillis();

        double fairObj = computeFairObjective(data, fairCenters);
        long t4 = System.currentTimeMillis();

        System.out.printf("Fair Objective with Standard Centers = %.4f\n", stdObj);
        System.out.printf("Fair Objective with Fair Centers = %.4f\n", fairObj);
        System.out.printf("Time to compute standard centers = %d ms\n", (t1 - t0));
        System.out.printf("Time to compute fair centers = %d ms\n", (t2 - t1));
        System.out.printf("Time to compute objective with standard centers = %d ms\n", (t3 - t2));
        System.out.printf("Time to compute objective with fair centers = %d ms\n", (t4 - t3));

        sc.stop();
    }

    // ---- Utility Methods ----

    public static double squaredDistance(Vector v1, Vector v2) {
        double sum = 0;
        for (int i = 0; i < v1.size(); i++) {
            double diff = v1.apply(i) - v2.apply(i);
            sum += diff * diff;
        }
        return sum;
    }

    public static Map<Integer, Tuple2<Double, double[]>> gatherAndReduce(JavaRDD<PointWithGroup> rdd, Vector[] centers, int dim) {
        return rdd.mapToPair(p -> {
            int best = -1;
            double bestDist = Double.MAX_VALUE;
            for (int i = 0; i < centers.length; i++) {
                double dist = squaredDistance(p.point, centers[i]);
                if (dist < bestDist) {
                    bestDist = dist;
                    best = i;
                }
            }
            return new Tuple2<>(best, p.point.toArray());
        }).groupByKey().mapValues(pts -> {
            int count = 0;
            double[] sum = new double[dim];
            for (double[] pt : pts) {
                for (int j = 0; j < dim; j++) sum[j] += pt[j];
                count++;
            }
            return new Tuple2<>((double) count, sum);
        }).collectAsMap();
    }

    public static double[] scaleVector(double[] vec, double scale) {
        return Arrays.stream(vec).map(x -> x * scale).toArray();
    }

    public static double[] diff(double[] a, double[] b) {
        double[] res = new double[a.length];
        for (int i = 0; i < a.length; i++) res[i] = a[i] - b[i];
        return res;
    }

    public static double norm(double[] v) {
        return Math.sqrt(Arrays.stream(v).map(x -> x * x).sum());
    }

    public static double[] normalize(double[] arr, long total) {
        return Arrays.stream(arr).map(x -> x / total).toArray();
    }

    public static double[] blend(double[] a, double[] b, double x, double l) {
        double[] result = new double[a.length];
        for (int i = 0; i < a.length; i++) {
            result[i] = ((l - x) * a[i] + x * b[i]) / l;
        }
        return result;
    }

    public static Vector[] toVectors(double[][] arr) {
        return Arrays.stream(arr).map(Vectors::dense).toArray(Vector[]::new);
    }

    // computeVectorX must be implemented elsewhere and available statically
}
