package com.lg.sort;

/**
 * Created by lg on 2017/11/20.
 * 优先队列实现的堆排序
 */
public class MaxPQ<Key extends Comparable<Key>> {
    private Key[] pq;
    private int N = 0;

    public MaxPQ(int max) {
        pq = (Key[]) new Comparable[max + 1];
    }

    boolean isEmpty() {
        return N == 0;
    }

    public int size() {
        return N;
    }

    public void insert(Key v) {
        pq[++N] = v;
        swim(N);
    }

    private void swim(int k) {
        while (k > 1 && less(k / 2, k)) {
            exch(k / 2, k);
            k = k / 2;
        }
    }

    public Key delMax() {
        Key max = pq[1]; //从根结点得到最大元素
        exch(1, N--); //交换其和最后一个节点
        pq[N + 1] = null;//防止对象游离
        sink(1);//恢复堆的有序性
        return max;
    }

    private void sink(int k) {
        while (2 * k <= N) {
            int j = 2 * k;
            if (j < N && less(j, j + 1)) j++;
            if (!less(k, j)) break;
            exch(k, j);
            k = j;
        }
    }

    private boolean less(int i, int j) {
        return pq[i].compareTo(pq[j]) < 0;
    }

    private void exch(int i, int j) {
        Key t = pq[i];
        pq[i] = pq[j];
        pq[j] = t;
    }

    private static void show(MaxPQ<String> a) {
        while(!a.isEmpty()){
            System.out.println(a.delMax());
        }
        System.out.println();
    }

    public static void main(String[] args) {
        int N=5;
        String[] b = {"a","g", "e", "z", "d"};
        MaxPQ<String> a = new MaxPQ<String>(N);
        for(int i=0;i<N;i++){
            a.insert(b[i]);
        }
        show(a);
    }


}
