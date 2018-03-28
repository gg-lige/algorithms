package com.lg.offer;

import java.text.ParseException;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Calendar;
import java.util.Date;
import java.util.List;
import java.util.Scanner;

public class Ali {

	public static void main(String[] args) throws ParseException {
		Scanner in = new Scanner(System.in);
		String[] res;

		String _repay_start_day;
		try {
			_repay_start_day = in.nextLine();
		} catch (Exception e) {
			_repay_start_day = null;
		}

		res = replay_plan(_repay_start_day);
		for (int res_i = 0; res_i < res.length; res_i++) {
			System.out.println(String.valueOf(res[res_i]));
		}

	}

	private static String[] replay_plan(String repay_start_day) throws ParseException {
		SimpleDateFormat dateFormat = new SimpleDateFormat("yyyyMMdd");
		Date originaldate = (Date) dateFormat.parse(repay_start_day);
		List<String> strings = new ArrayList<>();
		int time = 1;
		for (int i = 0; i < 8; i++) {
			for (int j = 0; j < 8; j++) {
				int credit = 0;
				// r1
				int temp = i;
				while (temp > 0) {
					credit++;
					temp = (temp - 1) & temp;
				}
				temp = j;
				while (temp > 0) {
					credit++;
					temp = (temp - 1) & temp;
				}

				// r2
				temp = i & j;
				while (temp > 0) {
					credit++;
					temp = (temp - 1) & temp;
				}

				// r3
				temp = i | j;
				if (temp == 7) {
					credit++;
				}

				if (credit <= 3) {
					StringBuffer sbBuffer = new StringBuffer();
					int s = 4;
					sbBuffer.append("情况" + time + ":");		
					Date date =originaldate;
					Calendar cd = Calendar.getInstance(); 
					while (s > 0) {

						if ((s & i) > 0) {
							sbBuffer.append(DateToStr(date) + "-" + "小波(异常),");
						} else {
							sbBuffer.append(DateToStr(date) + "-" + "小波(正常),");
						}

						if ((s & j) > 0) {
							sbBuffer.append(DateToStr(date) + "-" + "小钱(异常),");
						} else {
							sbBuffer.append(DateToStr(date) + "-" + "小钱(正常),");
						}
						s >>= 1;
						
						cd.setTime(date);
						cd.add(Calendar.DATE, 1);
						date=cd.getTime();

					}
					strings.add(sbBuffer.toString().substring(0, sbBuffer.length() - 1));
					time++;

				}

			}

		}
		String[] reStrings = new String[strings.size()];
		for (int c = 0; c < strings.size(); c++) {
			reStrings[c] = strings.get(c);
		}

		return reStrings;
	}

	private static String DateToStr(Date date) {
		SimpleDateFormat format = new SimpleDateFormat("yyyyMMdd");
		String str = format.format(date);
		return str;
	}

}
