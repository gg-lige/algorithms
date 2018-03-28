package com.lg.sort;

/**
 * Created by lg on 2017/11/15.
 */
public class Heap {
    /**
     * sort与sink函数的a数组的值在1-N之间
     * @param a
     */
    public static void sort(Comparable[] a) {
        int N = a.length;
        for( int k=N/2;k>=1;k--)
            sink(a,k,N);
        while(N>1){
            exch(a,1,N--);
            sink(a,1,N);
        }
    }

    private static void sink(Comparable[] a, int k, int N) {
        while (2 * k <= N) {
            int j = 2 * k;
            if (j < N && less(a,j, j + 1)) j++;
            if (!less(a,k, j)) break;
            exch(a,k, j);
            k = j;
        }
    }

/*
注意less 与exch函数的索引值均-1，

 */
    private static boolean less(Comparable[] a,int i, int j) {
        return a[i-1].compareTo(a[j-1]) < 0;
    }

    private static void exch(Comparable[] a, int i, int j) {
        Comparable t = a[i-1];
        a[i-1] = a[j-1];
        a[j-1] = t;
    }

    private static void show(Comparable[] a) {
        for (int i = 0; i < a.length; i++) {
            System.out.println(a[i]);
        }
        System.out.println();
    }



    public static void main(String[] args) {
        String[] a = {"a", "g", "e", "z", "d"};
        sort(a);
        show(a);
    }

}
