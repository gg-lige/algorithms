package com.lg.graph;


import java.io.IOException;

/**
 * Created by lg on 2017/11/28.
 *
 * 有向图
 *
 */
public class Digraph {
    private final int V;//顶点数目
    private int E;
    private Bag<Integer>[] adj; //邻接表

    public Digraph(int V) {
        this.V = V;
        this.E = 0;
        adj = (Bag<Integer>[]) new Bag[V];
        for (int v = 0; v < V; v++) {
            adj[v] = new Bag<Integer>();
        }
    }

    public Digraph(In in) throws IOException {
        this(in.readInt()); //读取v并初始化
        int E = in.readInt();  //读取E
        for (int i = 0; i < E; i++) {
            int v = in.readInt();  //读取一个顶点
            int w = in.readInt();    //读取另一个顶点
            addEdge(v, w);  //添加边
        }
    }



    public int V() {
        return V;
    }

    public int E() {
        return E;
    }

    public void addEdge(int v, int w) {
        adj[v].add(w);
        E++;
    }

    public Iterable<Integer> adj(int v) {
        return adj[v];
    }

    public Digraph reverse(){
        Digraph R =new Digraph(V);
        for(int v=0;v<V;v++)
            for( int w:R.adj(v))
                R.addEdge(w,v);

        return  R;

    }
}
