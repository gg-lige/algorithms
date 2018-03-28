package com.lg.search;

/**
 * Created by lg on 2017/11/26.
 */
public class RedBlackBST<Key extends Comparable<Key>,Value> {
    private Node root;
    private static final boolean RED= true;
    private static final boolean BLACK= false;

    private class Node{
        Key key;
        Value val;
        Node right,left;
        int N;
        boolean color;

        Node(Key key,Value val,int N,boolean color){
            this.key=key;
            this.val=val;
            this.N=N;
            this.color=color;
        }
    }

    private boolean isRed(Node x){
        if(x==null) return false;
        return x.color==RED;
    }

    private Node rotateLeft(Node h){
        Node x=h.right;
        h.right=x.left;
        x.left=h;
        x.color=h.color;
        h.color=RED;
        x.N=h.N;
        h.N=1+size(h.left)+size(h.right);
        return x;
    }

    private Node rotateRight(Node h){
        Node x=h.left;
        h.left=x.right;
        x.right=h;
        x.color=h.color;
        h.color=RED;
        x.N=h.N;
        h.N=1+size(h.left)+size(h.right);
        return x;
    }
    public int size() {
        return size(root);
    }

    private int size(Node x) {
        if (x == null) return 0;
        else return x.N;
    }
    private void flipColors(Node h){
        h.color=RED;
        h.left.color=BLACK;
        h.right.color=BLACK;
    }
    public void put(Key key,Value val){
        root=put(root,key,val);
        root.color=BLACK;
    }

    private Node put(Node h,Key key,Value val){
        if(h==null)
            return new Node(key,val,1,RED);
        int cmp=key.compareTo(h.key);
        if(cmp<0) h.left=put(h.left,key,val);
        else if(cmp>0) h.right=put(h.right,key,val);
        else h.val=val;

        if(isRed(h.right)&&!isRed(h.left)) h=rotateLeft(h);
        if(isRed(h.left)&&isRed(h.left.left)) h=rotateRight(h);
        if(isRed(h.left)&&isRed(h.right)) flipColors(h);

        h.N=size(h.left)+size(h.right)+1;
        return h;
     }

    public void deleteMin(){
        if(!isRed(root.left)&&!isRed(root.right))
            root.color=RED;

        root=deleteMin(root);
        if(!isEmpty())
            root.color=BLACK;

    }

    private Node deleteMin(Node h) {
        if(h.left==null)
            return null;
        
        if(!isRed(h.left)&&!isRed(h.left.left))
            h=moveRedLeft(h);
        
        h.left=deleteMin(h.left);
        return balance(h);
        
    }

    private Node balance(Node h) {
        if (isRed(h.right)) h = rotateLeft(h);
        if (isRed(h.left) && isRed(h.left.left)) h = rotateRight(h);
        if (isRed(h.left) && isRed(h.right)) flipColors(h);

        h.N = size(h.left) + size(h.right) + 1;
        return h;
    }

    private Node moveRedLeft(Node h) {
        flipColors(h);
        if (isRed(h.right.left)) {
            h.right = rotateRight(h.right);
            h = rotateLeft(h);
        }
        return h;
    }

    public boolean isEmpty() {
        return root == null;
    }


    public Key min() {
        return min(root).key;
    }

    private Node min(Node x) {
        if (x.left == null) return x;
        return min(x.left);
    }

    public Key max() {
        return max(root).key;
    }

    private Node max(Node x) {
        if (x.right == null) return x;
        return max(x.right);
    }


    public Iterable<Key> keys() {
        return keys(min(), max());
    }

    private Iterable<Key> keys(Key lo, Key hi) {
        Queue<Key> queue = new Queue<>();
        keys(root, queue, lo, hi);
        return queue;
    }

    private void keys(Node x, Queue<Key> queue, Key lo, Key hi) {
        if (x == null) return;
        int cmplo = lo.compareTo(x.key);
        int cmphi = hi.compareTo(x.key);
        if (cmplo < 0) keys(x.left, queue, lo, hi);
        if (cmplo <= 0 && cmphi >= 0) queue.enqueue(x.key);
        if (cmphi > 0) keys(x.right, queue, lo, hi);
    }
    public Value get(Key key) {
        return get(root, key);
    }

    private Value get(Node x, Key key) {
        //在以x为根结点的子树中，返回key对应的值
        if (x == null) return null;
        int cmp = key.compareTo(x.key);

        if (cmp > 0) return get(x.right, key);
        else if (cmp < 0) return get(x.left, key);
        else return x.val;
    }


    public boolean contains(Key s) {
        return get(s) != null;
    }
}
