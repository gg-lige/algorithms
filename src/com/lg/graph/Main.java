package com.lg.graph;

/**
 * Created by lg on 2018/3/13.
 */
import java.io.*;
import java.util.*;
import java.text.*;
import java.math.*;
import java.util.regex.*;

public class Main {

/** 请完成下面这个函数，实现题目要求的功能 **/
    /** 当然，你也可以不按照这个模板来作答，完全按照自己的想法来 ^-^  **/
    static String[] replay_plan(String repay_start_day) throws  ParseException{
        SimpleDateFormat format = new SimpleDateFormat("yyyyMMdd");
        Date date = format.parse(repay_start_day);

        List<String> list = new ArrayList<>();

        for (int i = 0; i <= 7; i++) {
            for (int j = 0; j <= 7; j++) {
                int score = 0;
                // rule1
                int t = i;
                while (t > 0) {
                    score += t & 1;
                    t >>= 1;
                }
                t = j;
                while (t > 0) {
                    score += t & 1;
                    t >>= 1;
                }
                // rule2
                t = i & j;
                while (t > 0) {
                    score += t & 1;
                    t >>= 1;
                }
                // rule3
                t = i | j;
                if (t == 7) score++;

                if (score <= 3) {
                    StringBuilder sb = new StringBuilder();

                    // parse
                    int p = 4;
                    while (p > 0) {
                        if ((i & p) > 0) {
                            sb.append(DateToStr(date) + "-" + "小波(异常),");
                        }else {
                            sb.append(DateToStr(date) + "-" + "小波(正常),");
                        }
                        if ((j & p) > 0) {
                            sb.append(DateToStr(date) + "-" + "小钱(异常),");
                        }else {
                            sb.append(DateToStr(date) + "-" + "小钱(正常),");
                        }
                        p >>= 1;
                    }

                    list.add(sb.toString().substring(0, sb.length() - 1));
                }
            }
        }

        String[] res = new String[list.size()];
        for (int i = 0; i < list.size(); i++) {
            res[i] = list.get(i);
        }
        return  res;
    }

    public static String DateToStr(Date date) {

        SimpleDateFormat format = new SimpleDateFormat("yyyyMMdd");
        String str = format.format(date);
        return str;
    }

    public static void main(String[] args) throws ParseException{
        Scanner in = new Scanner(System.in);
        String[] res;

        String _repay_start_day;
        try {
            _repay_start_day = in.nextLine();
        } catch (Exception e) {
            _repay_start_day = null;
        }

        res = replay_plan(_repay_start_day);
        for(int res_i=0; res_i < res.length; res_i++) {
            System.out.println(String.valueOf(res[res_i]));
        }

    }
}