package com.lg.search;

/**
 * CreatedData by lg on 2017/11/27.
 */
public class LinearProbingHashST<Key, Value> {
    private int N;//符号表中键值对的总数
    private int M = 16;//线性探测表的大小
    private Key[] keys;
    private Value[] vals;

    public LinearProbingHashST(int cap) {
        keys = (Key[]) new Object[M];
        vals = (Value[]) new Object[M];
        M = cap;
    }

    private int hash(Key key) {
        return (key.hashCode() & 0x7fffffff) % M;
    }


    public void put(Key key, Value val) {
        if (N >= M / 2) resize(2 * M);
        int i;

        for (i = hash(key); keys[i] != null; i = (i + 1) % M)
            if (keys[i].equals(key)) {
                vals[i] = val;
                return;
            }
        keys[i] = key;
        vals[i] = val;
        N++;
    }

    private void resize(int cap) {
        LinearProbingHashST<Key, Value> t;
        t = new LinearProbingHashST<Key, Value>(cap);
        for (int i = 0; i < M; i++)
            if (keys[i] != null)
                t.put(keys[i], vals[i]);
        keys = t.keys;
        vals = t.vals;

    }

    public Value get(Key key) {
        for (int i = hash(key); keys[i] != null; i = (i + 1) % M)
            if (keys[i].equals(key))
                return vals[i];
        return null;
    }

    public void delete(Key key) {
        if (key == null) throw new IllegalArgumentException("argument to delete() is null");
        if (!contains(key)) return;

        int i = hash(key);
        while (!key.equals(keys[i]))
            i = (i + 1) % M;
        keys[i] = null;
        vals[i] = null;
        i = (i + 1) % M;

        while (keys[i] != null) {
            Key keyRedoKey = keys[i];
            Value valReDoValue = vals[i];
            keys[i] = null;
            vals[i] = null;
            N--;
            put(keyRedoKey, valReDoValue);
            i = (i + 1) % M;
        }

        N--;
        if (N > 0 && N == M / 8) resize(M / 2);
    }

    public boolean contains(Key s) {
        return get(s) != null;
    }
}
