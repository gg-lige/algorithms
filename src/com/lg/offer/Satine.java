package com.lg.offer;

import java.util.Arrays;

public class Satine {
	// 1.确定字符互异
	public boolean checkDifferent(String iniString) {
		if (iniString.length() > 256)
			return false;

		boolean[] char_set = new boolean[65530];
		for (int i = 0; i < iniString.length(); i++) {
			int val = iniString.charAt(i);
			if (char_set[val])
				return false;
			char_set[val] = true;
		}
		return true;

	}

	// 2.原串翻转
	public String reverseString(String inString) {
		char[] iniString = inString.toCharArray();
		int i = 0;
		int j = iniString.length - 1;
		char c;
		while (i < j) {
			c = iniString[i];
			iniString[i] = iniString[j];
			iniString[j] = c;
			i++;
			j--;
		}
		return new String(iniString);
	}

	// 3.确定两串乱序同构
	public boolean checkSam(String stringA, String stringB) {
		if (stringA.length() != stringB.length())
			return false;
		return sort(stringA).equals(sort(stringB));
	}

	private String sort(String s) {
		char[] content = s.toCharArray(); // 将字符串转为字符数组的方式
		Arrays.sort(content);
		return new String(content);
	}

	// 5.基本字符串压缩
	public String zipString(String iniString) {
		// 检查压缩后的字符是否会变得更长
		int size = countZip(iniString);
		if (size > iniString.length())
			return iniString;

		// 压缩后长度变短的：
		StringBuffer mystr = new StringBuffer();
		char last = iniString.charAt(0);
		int count = 1;

		for (int i = 1; i < iniString.length(); i++) {
			if (iniString.charAt(i) == last)
				count++;
			else {
				mystr.append(last);
				mystr.append(count);
				last = iniString.charAt(i);
				count = 1;
			}
		}
		// 最后一组字符与数字并未存入
		mystr.append(last);
		mystr.append(count);
		return mystr.toString();
	}

	int countZip(String str) {
		if (str == null || str.isEmpty())
			return 0;
		char last = str.charAt(0);
		int count = 1;
		int size = 0;

		for (int i = 1; i < str.length(); i++) {
			if (str.charAt(i) == last)
				count++;
			else {
				last = str.charAt(i);
				size = size + 1 + String.valueOf(count).length();// 1为字母，后面为个数
				count = 1;
			}
		}
		size += 1 + String.valueOf(count).length();
		return size;
	}
	
	//6.像素翻转
    public int[][] transformImage(int[][] mat, int n) {
        // write code here
        for(int layer=0;layer<n/2;layer++){
            int first=layer;
            int last=n-1-layer;
            for(int i=first;i<last;i++){
                int offset= i-first;
                int top=mat[first][i];
                mat[first][i]=mat[last-offset][first];
                mat[last-offset][first]=mat[last][last-offset];
                mat[last][last-offset]=mat[i][last];
                mat[i][last]=top;
            }    
        }
        return mat;
    }
    
    //7.
    

}
