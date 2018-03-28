package com.lg.offer;

import java.util.ArrayList;
import java.util.List;

//当堆上没有足够的空间分配对象，又不能扩容时，就抛出OutofMemory
public class OutOfMemory {
	static class OOMObject{}
	
	public static void main(String[] args) {
		List<OOMObject> list= new ArrayList<OOMObject>();
		while(true){
			list.add(new OOMObject());
		}
	}
	
}
