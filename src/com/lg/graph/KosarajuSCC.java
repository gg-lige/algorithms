package com.lg.graph;

/**
 * Created by lg on 2017/12/14.
 * 计算强连通分量
 *
 */
public class KosarajuSCC {
    private boolean[] marked;  //已访问过的顶点
    private int[] id;  //强连通分量的标识符
    private int count;   //强连通分量的数量

    public KosarajuSCC(Digraph G) {
        marked = new boolean[G.V()];
        id = new int[G.V()];
        DepthFirstOrder order = new DepthFirstOrder(G.reverse());  //注意使用反向图
        for (int s:order.reversePost()) {  //使用反向图的逆后序拓扑作为深度优先搜索的遍历
            if (!marked[s]) {
                dfs(G, s);
                count++;
            }
        }

    }

    private void dfs(Digraph G, int v) {
        marked[v] = true;
        id[v] = count;
        for (int w : G.adj(v))
            if (!marked[w])
                dfs(G, w);
    }

    public boolean strongConnected(int v,int w){
        return id[v]==id[w];
    }

    public int id(int v){
        return id[v];
    }

    public int count(){
        return count;
    }



}
