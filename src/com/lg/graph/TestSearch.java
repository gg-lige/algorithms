package com.lg.graph;


import java.io.IOException;

/**
 * Created by lg on 2017/12/11.
 */
public class TestSearch {
    /*
        public static void main(String[] args) throws IOException {
            Graph G=new Graph(new In(args[0]));
            int s= Integer.parseInt(args[1]);
            DepthFirstPaths search= new DepthFirstPaths(G,s);
       //？     BreadthFirstPaths search= new BreadthFirstPaths(G,s);
              for(int v=0;v<G.V();v++){
                System.out.print(s+" to "+v+ ": ");
                if(search.hasPathTo(v))
                    for(int x:search.pathTo(v))
                        if(x==s) System.out.print(x);
                        else  System.out.print("-"+x);
                System.out.println();
            }


        }

       */


    public static void main(String[] args) throws IOException {
        Graph G = new Graph(new In(args[0]));
        CC cc = new CC(G);
        int M = cc.count();
        System.out.println(M + " componects");

        Bag<Integer>[] componects;
        componects=(Bag<Integer>[]) new Bag[M];
        for(int i=0;i<M;i++)
            componects[i]=new Bag<Integer>();
        for(int v=0;v<G.V();v++)
            componects[cc.id(v)].add(v);
        for(int i=0; i<M;i++){
            for(int v:componects[i])
                System.out.print(v+" ");
            System.out.println();

        }


    }

}
