package com.lg.company.qihu;

import com.lg.graph.In;

import java.util.Arrays;
import java.util.HashMap;
import java.util.Scanner;

/**
 * Created by lg on 2018/3/27.
 */
public class saima2 {

    public static void main(String[] args) {
        Scanner in = new Scanner(System.in);
        int n = in.nextInt(); //几组
        String[] result = new String[n];
        int j =0;
        while (j <n) {//注意while处理多个case
            int curSum =0;
            int n2 = in.nextInt();
            int[] v = new int[n2 * 2];

            for(int i =0;i< n2 *2;i++){
                v[i]=in.nextInt();
            }
          //  Arrays.sort(v);
            HashMap<Integer,Integer> map = new HashMap<>();
            for(int c=0;c< n;c++){
                if(!map.containsKey(v[c]))
                    map.put(v[c],1);
                else
                    map.replace(v[c],map.get(v[c])+1);
            }
            int count =0;
            for(Integer geshu:map.values()){
                if(geshu>=2)
                    count++;
            }
            if(count > n2){
                result[j]="YES";
            }
            else
                result[j] ="NO";
            j++;
        }

        for(int i =0;i< n ;i++){
            System.out.println(result[i]);
        }

    }

}
