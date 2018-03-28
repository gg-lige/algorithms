package com.lg.offer;

/**
 * Created by lg on 2018/3/26.
 */
public class TreeLinkNode {
    int val;
    TreeLinkNode left = null;
    TreeLinkNode right = null;
    TreeLinkNode next = null; //指向父节点

    TreeLinkNode(int val) {
        this.val = val;
    }
}
