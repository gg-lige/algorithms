package com.lg.meituan;

import java.util.Scanner;

/**
 * Created by lg on 2018/3/22.
 * 字符串距离
 */
public class StringDistance {

    public static void main(String[] args) {
        Scanner sc = new Scanner(System.in);
        char[] s = sc.nextLine().toCharArray();
        char[] t = sc.nextLine().toCharArray();

        int n = s.length;
        int m = t.length;

        if (n < m) {
            System.out.println("");
            return;
        }

        if (n == m) {
            int count = 0;
            for (int i = 0; i < n; i++) {
                if (s[i] != t[i]) count++;
            }
            System.out.println(count);
        } else {
            //n>m
            int count = 0;
            for (int i = 0; i <= n - m; i++) {
                for (int j = 0; j < m; j++) {
                    if (s[i+j] != t[j])
                        count++;
                }
            }
            System.out.println(count);
        }

    }


}
