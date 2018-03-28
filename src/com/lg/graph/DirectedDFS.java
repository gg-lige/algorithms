package com.lg.graph;

import java.io.IOException;

/**
 * Created by lg on 2017/12/13.
 *
 * 有向图的可达性
 *
 */
public class DirectedDFS {
    private boolean[] marked;

    public DirectedDFS(Digraph G,int s){
        marked= new boolean[G.V()];
        dfs(G,s);
    }

    public DirectedDFS(Digraph G,Iterable<Integer> sources){
        marked= new boolean[G.V()];
        for(int s:sources)
            if(!marked[s]) dfs(G,s);
    }

    private void dfs(Digraph G, int v) {
        marked[v]=true;
        for(int w:G.adj(v))
            if(!marked[w]) dfs(G,w);
    }

    public boolean marked(int v){
        return marked[v];
    }

    public static void main(String[] args) throws IOException {
        Digraph G= new Digraph(new In(args[0]));

        Bag<Integer> sources= new Bag<Integer>();
        for(int i=1;i<args.length;i++)
            sources.add(Integer.parseInt(args[i]));

        DirectedDFS reachable =new DirectedDFS(G,sources);

        for( int v=0;v<G.V();v++)
            if(reachable.marked(v))
                System.out.print(v+" ");
        System.out.println();


    }

}
