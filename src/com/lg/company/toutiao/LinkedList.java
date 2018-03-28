package com.lg.company.toutiao;

/**
 * Created by lg on 2018/3/22.
 */

/**
 * 单链表相关操作
 */
public class LinkedList<T> {
    class Node<T> {
        T val; //节点数据
        Node next; //指向的下一个节点

        public Node(T val, Node next) {
            this.val = val;
            this.next = next;
        }

        public T getVal() {
            return val;
        }

        public void setVal() {
            this.next = next;
        }

    }

    private Node first; //表头
    private Node tail; //表尾
    private int N;//个数

    public LinkedList() {
        first = null;
        tail = null;
    }

    public int size() {
        return N;
    }

    public boolean isEmpty() {
        return first == null;
    }

    public void clear() {
        first = null;
        tail = null;
        N = 0;
    }

    //头插入
    public void insertInHead(T value) {
        if (tail == null)   //若为空链表，让尾指针指向头指针
            tail = first;
        Node oldFirst = first; //保存指向链表的链接
        first = new Node<T>(value, oldFirst); //创建新的首节点
        N++;
    }

    //尾插入
    public void insertInTail(T value) {
        if (first == null) {  //若链表为空
            first = new Node(value, null);
            tail = first;
        } else {
            Node node = new Node<T>(value, null);
            tail.next = node;
            tail = node;//尾节点后移
        }
        N++;
    }

    //在指定位置插入元素
    public void insert(int index, T value) {
        if (index < 0 || index > N)
            System.out.println("索引下标越界");

        if (index == 0)
            insertInHead(value);
        else if (index > 0 && index < N - 1) {
            Node newFirst = first;
            for (int i = 0; i < index - 1; i++) {
                newFirst = newFirst.next;
            }
            Node insertNode = new Node<T>(value, newFirst.next);
            newFirst.next = insertNode;
            N++;
        } else
            insertInTail(value);

    }

    //删除指定位置的元素
    public void remove(int index) {
        if (index < 0 || index > N - 1){
            throw new IndexOutOfBoundsException("索引越界");

        }
        if(index == 0){ //flag<0说明删除的是第一个元素，将头结点指向下一个
            first = first.next;
        }else{
            Node pre =first;
            for(int i =0; i<index-1;i++){
                pre=pre.next;
            }
            Node cur =pre.next;
            pre.next=cur.next;
            if(index ==N-1){
                tail=pre;
            }
        }
        N--;
    }

    private Node getNodeByIndex(int index) {
        if(index<0||index>N-1){
            System.out.println("索引越界");
        }
        Node newFirst = first;
        for (int i = 0; i < index - 1; i++) {
            newFirst = newFirst.next;
        }
        return newFirst;
    }

    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder();
        Node node = first;
        for (int i = 0; i < N; i++, node = node.next) {
            sb = sb.append(node.getVal() + " ");
        }
        return sb + "";
    }

    public static void main(String[] args){
        LinkedList linklist = new LinkedList<>();
        linklist.insertInHead(2);
        linklist.insertInHead(3);
        linklist.insertInHead(6);
        linklist.insertInHead(8);
        linklist.insert(2,100);
        //   linklist.remove(1);
        linklist.remove(4);
        System.out.println(linklist);

    }

}
