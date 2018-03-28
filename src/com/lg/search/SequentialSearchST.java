package com.lg.search;


/**
 * Created by lg on 2017/11/22.
 */
public class SequentialSearchST<Key, Value> {
    private Node first;

    private class Node {
        Key key;
        Value val;
        Node next;

        public Node(Key key, Value value, Node next) {
            this.key = key;
            this.val = value;
            this.next = next;
        }
    }

    public boolean contains(Key s) {
        return get(s) != null;
    }

    public Iterable<Key> keys() {
        Queue<Key> queue = new Queue<Key>();
        for (Node p = first; p != null; p = p.next) {
            queue.enqueue(p.key);
        }
        return queue;
    }

    public void put(Key key, Value value) {
        for (Node x = first; x != null; x = x.next)
            if (key.equals(x.key)) {
                x.val = value;  //命中，更新
                return;
            }
        first = new Node(key, value, first);  //未命中，新建节点
    }

    public Value get(Key key) {
        for (Node x = first; x != null; x = x.next)
            if (key.equals(x.key))
                return x.val;
        return null;
    }


}
