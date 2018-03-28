package com.lg.search;

/**
 * Created by lg on 2017/11/23.
 */
public class BinarySearchST<Key extends Comparable<Key>,Value>{
    private Key[] keys;
    private Value[] values;
    private int N;//有序数组的元素个数

    public BinarySearchST( int capacity){
        keys=(Key[]) new Comparable[capacity];
        values=(Value[]) new Object[capacity];
    }

    public int size(){
        return N;
    }
    public boolean isEmpty(){
        return N==0;
    }

    public Value get(Key key){
        if(isEmpty()) return null;
        int i=rank(key);
        if(i<N &&keys[i].compareTo(key)==0) return values[i];
        else return null;
    }

    public void put(Key key,Value val){
        int i =rank(key);
        if(i<N &&keys[i].compareTo(key)==0) {
            values[i] = val;
            return;
        }
        for(int j=N;j>i;j--){
            keys[j]=keys[j-1];
            values[j]=values[j-1];
        }
        values[i]=val;
        keys[i]=key;
        N++;
    }

    public int rank(Key key){
        int lo=0,hi=N-1;
        while(lo<=hi){
            int mid=lo+(hi-lo)/2;
            int cmp=key.compareTo(keys[mid]);
            if(cmp<0) hi=mid-1;
            else if(cmp>0) lo=mid+1;
            else return mid;
        }
        return lo;
    }

    public Key min(){
        return keys[0];
    }
    public Key max(){
        return keys[N-1];
    }
    public Key select( int k){
        return keys[k];
    }
    public Key ceiling( Key key){
        int i=rank(key);
        return keys[i];
    }

    public Key floor( Key key){
        int i=rank(key);
        return keys[i-1];
    }

    public Iterable<Key>  keys(Key lo,Key hi){
        Queue<Key> queue=new Queue<Key>();
        for(int i=rank(lo);i<rank(hi);i++){
            queue.enqueue(keys[i]);
        }
        if(contains(hi))
            queue.enqueue(keys[rank(hi)]);
        return queue;
    }

    public boolean contains(Key s) {
        return get(s) != null;
    }


    public void delete(Key key){
        int i=rank(key);
        while(i<N){
            keys[i]=keys[i+1];
            values[i]=values[i+1];
        }
        N--;
        keys[N]=null;
        values[N]=null;
    }
}
