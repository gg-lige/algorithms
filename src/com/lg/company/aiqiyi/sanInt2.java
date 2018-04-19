package com.lg.company.aiqiyi;

import java.util.HashMap;
import java.util.Scanner;

/**
 * Created by lg on 2018/3/27.
 */
public class sanInt2 {

    public static void main(String[] args) {
        Scanner scan = new Scanner(System.in);
        int a = scan.nextInt();
        int b = scan.nextInt();
        int c = scan.nextInt();
        int max = Math.max(Math.max(a,b),c);
        int dif = (3*max - a - b - c);
        if(dif % 2 == 0) System.out.println(dif/2);
        else System.out.println(dif/2 +2);
    }


}
