package com.lg.sort;

import java.util.Random;

/**
 * Created by lg on 2017/11/16.
 */
public class SortCompare {
    public static double time(String alg, Comparable[] a) {
        Stopwatch timer = new Stopwatch();
        if (alg.equals("Insertion")) Insertion.sort(a);
        if (alg.equals("Selection")) Selection.sort(a);
        return timer.elapsedTime();
    }

    //输入模型
    public static double timeRandomInput(String alg, int N, int T) {
        double total = 0D;
        Double[] a = new Double[N];
        Random r = new Random();
        for (int t = 0; t < T; t++) {
            for (int i = 0; i < N; i++)
                a[i] = r.nextDouble();
            total += time(alg, a);
        }
        return total;
    }

    public static void main(String[] args) {
        String arg1 = args[0];
        String arg2 = args[1];
        int N = Integer.parseInt(args[2]);
        int T = Integer.parseInt(args[3]);
        double t1=timeRandomInput(arg1,N,T);
        double t2=timeRandomInput(arg2,N,T);
        System.out.printf("For %d random Doubles\n     %s is ",N,arg1);
        System.out.printf("%.1f times faster than %s\n",t2/t1,arg2);


    }

}

class Stopwatch {
    private final long start;

    public Stopwatch() {
        start = System.currentTimeMillis();
    }

    public double elapsedTime() {
        long now = System.currentTimeMillis();
        return (now - start) / 1000.0;
    }
}