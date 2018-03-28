package com.lg.offer;

/**
 * Created by lg on 2018/3/20.
 */

public class ListNodeV {
    int val;
    ListNodeV next;

    public ListNodeV(int val) {
        this.val = val;
    }
    private ListNodeV first =null;

    public void insert(int a){
        ListNodeV newnode= new ListNodeV(a);
        if(first==null){
            first=newnode;
            return;
        }
        newnode.next=first;
        first=newnode;

    }

    public void delete(int index){
        int i=1;
        ListNodeV preNode =first;
        ListNodeV curNode=preNode.next;
        while(curNode!=null){
            if(i==index){
                preNode.next=curNode.next;
            }
            preNode =curNode;
            curNode =curNode.next;
            i++;
        }
    }

}