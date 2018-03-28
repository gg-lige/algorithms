package com.lg.link;

/**
 * Created by lg on 2018/3/22.
 */

import java.util.Scanner;

class AsOrderList {
    class Node {
        int val;
        Node next;

        public Node(int val) {
            this.val = val;
        }
    }

    Node first;
    int N;

    public AsOrderList() {
        first = null;
    }


    public void insertHead(int val) {
        Node oldFirst = first;
        first = new Node(val);
        first.next = oldFirst;
        N++;
    }

    public String print() {
        StringBuilder sb = new StringBuilder();
        Node node = first;
        for (int i = 0; i < N; i++, node = node.next) {
            sb.append(node.val + " ");
        }
        sb.substring(0,sb.length()-1);
        return sb.toString();
    }

}

public class Main {
    public static void sort(int[] array, int N) {
        for (int i = 0; i < N; i++) {
            for (int j = i; j > 0 && array[j] < array[j - 1]; j--) {
                int temp = array[j];
                array[j] = array[j - 1];
                array[j - 1] = temp;
            }
        }
    }

    public static void main(String[] args) {
        Scanner sc = new Scanner(System.in);
        int N = sc.nextInt();
        int[] array = new int[N];

        for (int i = 0; i < N; i++) {
            array[i] = sc.nextInt();
        }
        sort(array, N);
        AsOrderList aList = new AsOrderList();

        for (int i = N - 1; i >= 0; i--) {
            aList.insertHead(array[i]);
        }
        System.out.print(aList.print());
    }


}