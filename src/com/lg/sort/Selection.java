package com.lg.sort;

import java.util.Scanner;

/**
 * Created by lg on 2017/11/15.
 */
public class Selection {
    public static void sort(Comparable[] a) {
        int N = a.length;
        for (int i = 0; i < N; i++) {
           int min=i;
           for(int j=i+1;j<N;j++){
               if(less(a[j],a[min]))
                   min=j;
           }
           exch(a,i,min);
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
        System.out.println("Please enter the number:");
        String[] a = new String[5];
        Scanner sc = new Scanner(System.in);
        for (int i = 0; i < a.length; i++) {
            a[i] = sc.next();
        }
        sort(a);
        assert isSorted(a);
        show(a);
    }

}
