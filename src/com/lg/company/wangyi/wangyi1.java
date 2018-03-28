package com.lg.wangyi;

import java.util.Scanner;

/**
 * Created by lg on 2018/3/27.
 */
public class wangyi1 {

    public static void main(String[] args) {
        Scanner in = new Scanner(System.in);
        while (in.hasNextInt()) {//注意while处理多个case
            int n = in.nextInt();
            int k = in.nextInt();
            int count =0;
            for(int i =1;i<=n;i++){
                for(int j =1;j<=n;j++){
                    if(i%j>=k)
                        count++;
                }
            }
            System.out.println(count);



        }
    }

}
