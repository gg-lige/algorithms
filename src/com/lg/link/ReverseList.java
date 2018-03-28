package com.lg.link;

/**
 * Created by yzk on 2018/3/13.
 *
 * ·´×ªÁ´±í
 */
public class ReverseList {
    class Node {
        int val;
        Node next;

        Node(int val) {
            this.val = val;
        }
    }

    private Node init(int n) {
        if (n < 0) return null;
        Node node = new Node(n);
        while (n > 0) {
            Node t = new Node(--n);
            t.next = node;
            node = t;
        }
        return node;
    }

    private void print(Node head) {
        Node current = head;
        while (current != null) {
            System.out.print(current.val + " ");
            current = current.next;
        }
        System.out.println();
    }



    private Node reverse1(Node head) {
        if (head == null || head.next == null) return head;

        Node prev = null;
        Node next = null;
        while (head != null) {
            next = head.next;
            head.next = prev;
            prev = head;
            head = next;
        }
        return prev;
    }

    private Node reverse2(Node head) {
        if (head == null || head.next == null) return head;

        Node node = reverse2(head.next);
        head.next.next = head;
        head.next = null;
        return node;
    }

    public static void main(String[] args) {
        ReverseList rl = new ReverseList();
        Node head = rl.init(5);
        rl.print(head);

//        rl.print(rl.reverse1(head));
        rl.print(rl.reverse2(rl.reverse1(head)));
    }

}



