package com.lg.graph;

import java.util.Stack;

/**
 * Created by lg on 2017/12/11.
 */
public class DepthFirstPaths {
    private boolean[] marked; //描述该定点是否调用过dfs()
    private int[] edgeTo;   //从起点到一个顶点的已知路径上的最后一个顶点
    private final int s;

    public DepthFirstPaths(Graph G, int s) {
        marked = new boolean[G.V()];
        edgeTo = new int[G.V()];
        this.s = s;
        dfs(G, s);

    }

    private void dfs(Graph G, int v) {
        marked[v] = true;
        for (int w : G.adj(v))
            if (!marked[w]) {
                edgeTo[w] = v;
                dfs(G, w);
            }
    }

    public boolean hasPathTo(int v) {
        return marked[v];
    }

    public Iterable<Integer> pathTo(int v) {
        if (!hasPathTo(v)) return null;
        Stack<Integer> path = new Stack<Integer>();     //深度优先搜索中注意使用栈来保存路径
        for (int x = v; x != s; x = edgeTo[x])
            path.push(x);
        path.push(s);
        return path;

    }


}
