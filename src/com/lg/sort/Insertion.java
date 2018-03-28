package com.lg.sort;

import java.util.Scanner;

/**
 * Created by lg on 2017/11/15.
 */
public class Insertion {
    public static void sort(Comparable[] a) {
        int N=a.length;
        for(int i=1;i<N;i++){
            for(int j=i;j>0&& less(a[j], a[j-1]);j--)
                exch(a,j-1,j);
        }
    }

    private static boolean less(Comparable a, Comparable b) {
        return a.compareTo(b) < 0;
    }

    private static void exch(Comparable[] a, int i, int j) {
        Comparable t = a[i];
        a[i] = a[j];
        a[j] = t;
    }

    private static void show(Comparable[] a) {
        for (int i = 0; i < a.length; i++) {
            System.out.println(a[i]);
        }
        System.out.println();
    }

    private static boolean isSorted(Comparable[] a) {
        for (int i = 1; i < a.length; i++) {
            if (less(a[i], a[i - 1]))
                return false;
        }
        return true;
    }

    public static void main(String[] args) {
        String[] a = {"a","g", "e", "z", "d"};
        sort(a);
        assert isSorted(a);
        show(a);
    }


}


