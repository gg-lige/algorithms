package com.lg.graph;

import com.lg.search.Queue;

import java.util.Stack;

/**
 * Created by lg on 2017/12/11.
 */
public class BreadthFirstPaths {
    private boolean[] marked; //描述该定点是否调用过dfs()
    private int[] edgeTo;   //从起点到一个顶点的已知路径上的最后一个顶点
    private final int s;

    public BreadthFirstPaths(Graph G,int s){
        marked=new boolean[G.V()];
        edgeTo= new int[G.V()];
        this.s=s;
        bfs(G,s);
    }

    private void bfs(Graph G, int s) {
        Queue<Integer> queue= new Queue<Integer>();      //广度优先搜索中注意使用队列
        marked[s]=true; // 标记起点
        queue.enqueue(s); //将其加入起点
        while(!queue.isEmpty()){
            int v= queue.dequeue(); //从队列中删去下一顶点
            for(int w: G.adj(v)){
                edgeTo[w]=v; //保存最短路径的最后一条边
                marked[w]=true;  //标记，因为最短路径已知
                queue.enqueue(w);  //并将其添加到队列
            }
        }
    }
    public boolean hasPathTo(int v){
        return marked[v];
    }

    public Iterable<Integer> pathTo(int v){
        if(!hasPathTo(v)) return null;
        Stack<Integer> path = new Stack<Integer>();     //深度优先搜索中注意使用栈来保存路径
        for(int x=v;x!=s;x=edgeTo[x])
            path.push(x);
        path.push(s);
        return path;

    }



}
