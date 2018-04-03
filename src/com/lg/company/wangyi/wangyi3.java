package com.lg.wangyi;

import java.util.Scanner;

/**
 * Created by lg on 2018/3/27.
 */
public class wangyi3 {

    public static void main(String[] args) {
        Scanner in = new Scanner(System.in);
        while (in.hasNextInt()) {//注意while处理多个case
            int n = in.nextInt();
            int w = in.nextInt();
            int[] v = new int[n]; //零食
            int[][] bestvalues = new int[n][w];
            for (int i = 0; i < n; i++) {
                v[i] = in.nextInt();
            }
            for (int j = 0; j <= w; j++) {
                for (int i = 0; i <= n; i++) {
                    if (i == 0 || j == 0) {
                        bestvalues[i][j] = 0;
                    } else {
                        if (j < v[i - 1]) {
                            bestvalues[i][j] = bestvalues[i - 1][j];
                        } else {
                            int weight = v[i - 1];
                       //     int value = n[i - 1];
                        }
                    }
                }
            }
     //       bestvalues = bestvalues[n][];
        }
    }

}
