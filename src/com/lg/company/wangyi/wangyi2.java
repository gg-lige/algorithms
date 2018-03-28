package com.lg.wangyi;

import java.util.Scanner;

/**
 * Created by lg on 2018/3/27.
 */
public class wangyi2 {

    public static void main(String[] args) {
        Scanner in = new Scanner(System.in);
        while (in.hasNextInt()) {//注意while处理多个case
            int n = in.nextInt();
            int initDirection = 0;//北方
            char[] dir = in.nextLine().toCharArray();
            for(int i=0;i<dir.length;i++){
                if(dir[i]=='L'){
                    initDirection--;
                }
                if(dir[i]=='R'){
                    initDirection++;
                }
            }
            if(initDirection%4==0){
                System.out.println("N");
            }
            if(initDirection%4==1){
                System.out.println("E");
            }
            if(initDirection%4==2){
                System.out.println("S");
            }
            if(initDirection%4==3){
                System.out.println("W");
            }

        }
    }

}
