package com.lg.offer;

public class EnumSingleton {
	public static void main( String[] args){
		danli one = danli.INSTANCE;
		danli two = danli.INSTANCE;
		System.out.println(one.getString());
		
		one.test();
		one.setString("I'm one!");
		System.out.println(one==two);
		System.out.println(two.getString());
		
	}
}


enum danli{
	INSTANCE;
	
	private String string=name();
	
	public String getString(){
		return string;
	}

	public void setString(String string) {
		this.string = string;
	}
	
	public void test(){
		System.out.println("teststart^^^^^^");
	}
	
	
}