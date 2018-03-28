package com.lg.offer;


/**
 * Created by lg on 2018/3/24.
 */

class Node {
    int val;
    int number;
    Node next;

    Node(int val, int number) {
        this.val = val;
        this.number = number;
    }

    @Override
    public String toString() {
        return this.val + "";
    }
}

public class Circle {
    public static void main(String[] args) {
        int n = 6;
        //先构造循环链表
        Node head = new Node(0,7); //头结点, 值为0
        Node pre = head;
        Node temp = null;
        for (int i = 1; i < n; i++) {
            temp = new Node(i,7);
            pre.next = temp;
            pre = temp;
        }
        temp.next = head;//将第n-1个结点(也就是尾结点)指向头结点
        Node temp2 = null;
        while (n != 1) {
            temp2 = head;
            //先找到第m个结点的前驱
            for (int i = 1; i < temp2.number - 1; i++) {
                temp2 = temp2.next;
            }
            //删除第m个结点：将第m个结点的前驱指向第m个结点后面那个结点,temp2表示第m个结点的前驱
            temp2.next = temp2.next.next;
            head = temp2.next; //更新头结点
            n--;
        }
        System.out.print(head.val);

    }
}
