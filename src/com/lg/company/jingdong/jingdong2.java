package com.lg.company.jingdong;

import java.util.Arrays;
import java.util.Collections;
import java.util.Scanner;

/**
 * Created by lg on 2018/4/9.
 */
public class jingdong2 {

    public static void main(String[] args) {
        Scanner in = new Scanner(System.in);
        int n = in.nextInt();
        String[] result = new String[n];
        int j = 0;
        while (j < n) {//注意while处理多个case
            int number = in.nextInt();

            StringBuilder sb = new StringBuilder();
            int n2 = n;
            while (n2 % 2 == 0 && n2 > 2) {
                n2 = n2 / 2;
            }
            int i = number / n2;
            if (((i & 1) == 1) && ((n2 & 1) == 0) || ((i & 1) == 0) && ((n2 & 1) == 1)) {
                sb.append(i + " " + n2);
            } else {
                sb.append("No");
            }

            result[j] = sb.toString();
            j++;
        }


        for (int i = 0; i < n; i++) {
            System.out.println(result[i]);
        }
    }


}
