package com.lg.meituan;

import java.util.Scanner;

/**
 * Created by lg on 2018/3/16.
 * 最长子串个数
 *
 */
public class LongestSubstring {
    public static void main(String[] args){
        String a="abcdef";
        String b="bacd";
        int len=longest(a,b);
        System.out.print(len);
    }

    private static int longest(String a, String b) {
        int maxlength=0;
        for(int i=0;i<a.length();i++){
            for(int j=i+1;j<a.length();j++){
                if(b.contains(a.substring(i,j)))
                    if(maxlength<j-i)
                        maxlength=j-i;

            }
        }
        return maxlength;
    }

}
