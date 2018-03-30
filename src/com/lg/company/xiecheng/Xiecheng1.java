package com.lg.company.xiecheng;

import java.util.Scanner;

/**
 * Created by lg on 2018/3/27.
 */
public class Xiecheng1 {


        static boolean check(int[] array) {
            Scanner sc = new Scanner(System.in);
            int nCount = sc.nextInt();
            int[] num = new int[nCount];
            int four=0 ,two=0;
            for(int i =0 ; i< nCount; i++){
                int x = sc.nextInt();
                if(x%4==0){
                    four++;
                }else if(x%2==0)
                    two++;
            }

            if(four>=(nCount-two+1)/2){
                System.out.print("1");
                return true;
            }else{
                System.out.print("0");
                return false;
            }




    }


    static int calculate(int[][] matrix) {
        Scanner sc = new Scanner(System.in);
        int row = sc.nextInt();
        int lie = sc.nextInt();
        int[][] num = new int[row][lie];

        for(int i =0; i< row; i++){
            for(int j =0;j<lie; j++){
                num[i][j]=sc.nextInt();
            }

        }
        int result =0;
        for()

    }

}
