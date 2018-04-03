package com.lg.company.qihu;

import java.util.Scanner;

/**
 * Created by lg on 2018/3/27.
 */
public class jiaoyi1 {

    public static void main(String[] args) {
        Scanner in = new Scanner(System.in);
        int n = in.nextInt();
        int friends =5;
        int[] result = new int[n];
        int j =0;
        while (j <n) {//注意while处理多个case
            int curSum =0;
            int[] money = new int[friends];
            for(int i =0;i< friends;i++){
                money[i]=in.nextInt();
                curSum += money[i];
            }
            if(curSum % 5 ==0){
                result[j]=curSum/5;
            }
            else
                result[j] =-1;
            j++;
        }
        for(int i =0;i< n ;i++){
            System.out.println(result[i]);
        }

    }

}
