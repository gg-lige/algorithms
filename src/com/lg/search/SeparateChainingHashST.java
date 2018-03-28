package com.lg.search;

/**
 * Created by lg on 2017/11/27.
 */
public class SeparateChainingHashST<Key, Value> {
    private int N; //键值对总数
    private int M; //散列表长度
    private SequentialSearchST<Key, Value>[] st;  //存放链表对象的数组

    public SeparateChainingHashST() {
        this(997);
    }

    public SeparateChainingHashST(int M) {
        this.M = M;//创建M条链表
        st = (SequentialSearchST<Key, Value>[]) new SequentialSearchST[M];
        for (int i = 0; i < M; i++)
            st[i] = new SequentialSearchST();
    }

    private int hash(Key key) {
        return (key.hashCode() & 0x7fffffff) % M;
    }

    public Value get(Key key){
        return (Value) st[hash(key)].get(key);
    }

    public void put(Key key,Value val){
        st[hash(key)].put(key,val);
    }
}
