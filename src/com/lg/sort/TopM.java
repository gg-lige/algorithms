package com.lg.sort;

import java.util.Scanner;
import java.util.Stack;

/**
 * Created by lg on 2017/11/20.
 */
public class TopM {


    public static void main(String[] args){
        int M =Integer.parseInt(args[0]);
        MaxPQ<String> pq= new MaxPQ<String>(M+1);
        Scanner sc = new Scanner(System.in);
        while(sc.hasNext()){
            pq.insert(new String(sc.nextLine()));
            if(pq.size()>M){
                pq.delMax();
            }
        }
        Stack<String> stack= new Stack();
        while(!pq.isEmpty())
            stack.push(pq.delMax());
        for(String s:stack)
            System.out.println(s);

    }
}
