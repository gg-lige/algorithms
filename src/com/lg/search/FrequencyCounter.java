package com.lg.search;

import org.junit.Test;

import java.io.*;

/**
 * Created by lg on 2017/11/22.
 */
public class FrequencyCounter {
    @Test
    public void testSequentialSearchST() throws IOException{
        int minlen= 2;
        SequentialSearchST<String, Integer> st= new SequentialSearchST<String, Integer>();
        String file= "E:\\IDEAworkspace\\algorithms\\src\\hello.txt";

        InputStream inputStream = new FileInputStream(file);
        InputStreamReader inputReader = new InputStreamReader(inputStream);
        BufferedReader bufferReader = new BufferedReader(inputReader);

        StringBuffer sb= new StringBuffer();
        String str=null;
        while((str=bufferReader.readLine())!=null){
            sb.append(str+" ");
        }
        System.out.println(sb);

        for(String s:sb.toString().split(" ")){
            if(s.length()<minlen)
                continue;
            if(!st.contains(s))
                st.put(s,1);
            else
                st.put(s,st.get(s)+1);
        }

        String max=" ";
        st.put(max,0);
        int c=0;
        for(String word:st.keys()) {
            if (st.get(word) > st.get(max))
                max = word;
            c++;
        }
        System.out.println(max+" "+st.get(max));

        inputStream.close();
        inputReader.close();
        bufferReader.close();
    }

    @Test
    public void testBinarySearchST() throws IOException{
        int minlen= 2;
        BinarySearchST<String, Integer> st= new BinarySearchST<String, Integer>(60);
        String file= "E:\\IDEAworkspace\\algorithms\\src\\hello.txt";

        InputStream inputStream = new FileInputStream(file);
        InputStreamReader inputReader = new InputStreamReader(inputStream);
        BufferedReader bufferReader = new BufferedReader(inputReader);

        StringBuffer sb= new StringBuffer();
        String str=null;
        while((str=bufferReader.readLine())!=null){
            sb.append(str+" ");
        }
        System.out.println(sb);

        for(String s:sb.toString().split(" ")){
            if(s.length()<minlen)
                continue;
            if(!st.contains(s))
                st.put(s,1);
            else
                st.put(s,st.get(s)+1);
        }


        String max=" ";
        st.put(max,0);
        for(String word:st.keys(st.min(),st.max())) {
            if (st.get(word) > st.get(max))
                max = word;
        }
        System.out.println(max+" "+st.get(max));

        inputStream.close();
        inputReader.close();
        bufferReader.close();
    }
    @Test
    public void testBST() throws IOException{
        int minlen= 2;
        BST<String, Integer> st= new BST<String, Integer>();
        String file= "E:\\IDEAworkspace\\algorithms\\src\\hello.txt";

        InputStream inputStream = new FileInputStream(file);
        InputStreamReader inputReader = new InputStreamReader(inputStream);
        BufferedReader bufferReader = new BufferedReader(inputReader);

        StringBuffer sb= new StringBuffer();
        String str=null;
        while((str=bufferReader.readLine())!=null){
            sb.append(str+" ");
        }
        System.out.println(sb);

        for(String s:sb.toString().split(" ")){
            if(s.length()<minlen)
                continue;
            if(!st.contains(s))
                st.put(s,1);
            else
                st.put(s,st.get(s)+1);
        }


        String max=" ";
        st.put(max,0);
        for(String word:st.keys()) {
            if (st.get(word) > st.get(max))
                max = word;
        }
        System.out.println(max+" "+st.get(max));

        inputStream.close();
        inputReader.close();
        bufferReader.close();
    }

    @Test
    public void testRedBlackBST() throws IOException{
        int minlen= 2;
        RedBlackBST<String, Integer> st= new RedBlackBST<String, Integer>();
        String file= "E:\\IDEAworkspace\\algorithms\\src\\hello.txt";

        InputStream inputStream = new FileInputStream(file);
        InputStreamReader inputReader = new InputStreamReader(inputStream);
        BufferedReader bufferReader = new BufferedReader(inputReader);

        StringBuffer sb= new StringBuffer();
        String str=null;
        while((str=bufferReader.readLine())!=null){
            sb.append(str+" ");
        }
        System.out.println(sb);

        for(String s:sb.toString().split(" ")){
            if(s.length()<minlen)
                continue;
            if(!st.contains(s))
                st.put(s,1);
            else
                st.put(s,st.get(s)+1);
        }


        String max=" ";
        st.put(max,0);
        for(String word:st.keys()) {
            if (st.get(word) > st.get(max))
                max = word;
        }
        System.out.println(max+" "+st.get(max));

        inputStream.close();
        inputReader.close();
        bufferReader.close();
    }


}
