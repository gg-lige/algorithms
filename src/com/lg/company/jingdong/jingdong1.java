package com.lg.company.jingdong;

import java.util.Scanner;

/**
 * Created by lg on 2018/4/9.
 */
public class jingdong1 {

    public static void main(String[] args) {
        Scanner in = new Scanner(System.in);
        String str = in.nextLine();
        int count =0;

        isHuiWen(str);
      //  System.out.print(result);

    }

    public static boolean isHuiWen(String text) {
        int length = text.length();
        for (int i = 0; i < length / 2; i++) {
            if (text.toCharArray()[i] != text.toCharArray()[length - i - 1]) {
                return false;
            }
        }
        return true;
    }
}
