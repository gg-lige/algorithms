package com.lg.company.tengxun;

import java.util.Scanner;

/**
 * Created by lg on 2018/3/27.
 */
public class tengxun2 {

    public static void main(String[] args) {
        Scanner in = new Scanner(System.in);
        int gedan = in.nextInt(); //歌单
        int l1 =in.nextInt() ;
        int n1 = in.nextInt();
        int l2 = in.nextInt();
        int n2 = in.nextInt();

        long count =0;
        for( int i =0 ;i <= gedan/l1; i++){
            for(int j =0; j <= Math.min((gedan- i*l1)/l2,n2); j++){
                if(i * l1 + j * l2 ==gedan){
                    long a =1;
                    for(int c=n1; c > n1-i;c--){
                        a *= c;
                    }
                    long b =1;
                    for(int c=i; c > 0;c--){
                        b *= c;
                    }
                    long a1=a/b;



                    long m =1;
                    for(int c=n2; c > n2-j;c--){
                        m *= c;
                    }
                    long n =1;
                    for(int c=j; c > 0;c--){
                        n *= c;
                    }
                    long a2=m/n;

                    count += a1* a2;
                }

            }
        }
        System.out.println(count%1000000007);
    }


}
