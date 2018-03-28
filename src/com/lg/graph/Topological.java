package com.lg.graph;

/**
 * Created by lg on 2017/12/14.
 * 拓扑排序，有三种
 *
 */
public class Topological {
    private Iterable<Integer> order;  //顶点的拓扑排序

    public Topological(Digraph G){
        DirectedCycle cyclefinder= new DirectedCycle(G);
        if(!cyclefinder.hasCycle()){   //当为有向无环图时才可构造拓扑排序
            DepthFirstOrder dfs = new DepthFirstOrder(G);
            order=dfs.reversePost();
        }
    }


    public Iterable<Integer> order(){
        return order;
    }

    public boolean isDAG(){
        return order!=null;
    }

    public static void main(String[] args){
        String filename= args[0];
        String separator= args[1];
        SymbolDiGraph sg= new SymbolDiGraph(filename,separator);
        Topological top =new Topological(sg.G());
        for(int v:top.order())
            System.out.println(sg.name(v));



    }

}
