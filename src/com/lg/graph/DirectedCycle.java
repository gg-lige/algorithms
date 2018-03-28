package com.lg.graph;

import java.util.Stack;

/**
 * Created by lg on 2017/12/14.
 *
 * 有向图中寻找环
 *
 */
public class DirectedCycle {
    private boolean[] marked;
    private int[] edgeTo;
    private Stack<Integer> cycle;   //有向环中的所有顶点
    private boolean[] onstack;  //递归调用的栈上的所有顶点

    public DirectedCycle(Digraph G) {
        onstack = new boolean[G.V()];
        edgeTo = new int[G.V()];
        marked = new boolean[G.V()];
        for (int v = 0; v < G.V(); v++)
            if (!marked[v]) dfs(G, v);


    }

    private void dfs(Digraph G, int v) {
        onstack[v] = true;
        marked[v] = true;
        for (int w : G.adj(v))
            if (this.hasCycle()) return;
            else if (!marked[w]) {
                edgeTo[w] = v;
                dfs(G, w);
            } else if (onstack[w]) {
                cycle = new Stack<Integer>();
                for (int x = v; x != w; x = edgeTo[x])
                    cycle.push(x);
                cycle.push(w);
                cycle.push(v);
            }
        onstack[v] = false;
    }

    public boolean hasCycle() {
        return cycle != null;
    }

    public Iterable<Integer> cycle(){
        return cycle;
    }

}
