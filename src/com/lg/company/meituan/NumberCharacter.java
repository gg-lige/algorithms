package com.lg.company.meituan;


import java.util.*;

/**
 * Created by lg on 2018/3/22.
 * 数字字符
 */
public class NumberCharacter {

    public int solveLength(String s) {
        int result = 1;
        Map<Integer, Integer> map = new HashMap<>();
        for (int a : s.toCharArray()) {
            if (map.containsKey(a)) {
                map.replace(a, map.get(a) + 1);
            } else {
                map.put(a, 1);
            }
        }
        return result;


    }

    public static void main(String[] args) {
        Scanner sc = new Scanner(System.in);

        char[] nums = sc.nextLine().toCharArray();
        int[] cnt = new int[10];
        for (int i = 0; i < nums.length; i++) {
            cnt[nums[i] - '0']++;
        }

        int c = 1;
        while (true) {
            for (int i = 1; i < cnt.length; i++) {
                if (cnt[i] < c) {
                    StringBuilder sb = new StringBuilder();
                    for (int j = 0; j < c; j++) {
                        sb.append(i);
                    }
                    System.out.println(sb.toString());
                    return;
                }
            }

            // 进位
            if (cnt[0] < c) {
                StringBuilder sb = new StringBuilder();
                sb.append(1);
                for (int i = 0; i < c; i++) {
                    sb.append(0);
                }
                System.out.println(sb.toString());
                return;
            }
            c++;
        }


    }


}
