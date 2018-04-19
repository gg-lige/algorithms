package com.lg.company.aiqiyi;

import java.util.Scanner;

/**
 * Created by lg on 2018/3/27.
 */
public class wanou1 {

    public static void main(String[] args) {
        Scanner in = new Scanner(System.in);
        int n = in.nextInt();
        Long[] a= new Long[n];
        for (int i = 0; i < n; i++) {
            a[i]= in.nextLong();

        }
        Long zhengque =a[0];

        if (zhengque != a[1]) {
            if(zhengque != a[2]){
                zhengque=a[1];
            }
        }

        for (int i = 0; i <n ; i++) {
            if(a[i]!= zhengque){
                System.out.println(i+1);
                break;
            }
        }


    }

}
