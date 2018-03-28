package com.lg.company.ali;

import java.util.Stack;

/**
 * Created by lg on 2018/3/25.
 *
 * ��һ��ջʵ����һ��ջ������
 */
public class SortStackByStack {

    public static void main(String[] args){
        Stack<Integer> stack= new Stack<>();
        stack.push(3);
        stack.push(7);
        stack.push(4);
        stack.push(8);
        stack.push(2);
        sortStackByStack(stack);
        System.out.print(stack);
    }

    private static void sortStackByStack(Stack<Integer> stack) {
        Stack<Integer> help= new Stack<>();
        while(!stack.isEmpty()){
            int cur=stack.pop();
            while(!help.isEmpty() && cur>help.peek()){
                stack.push(help.pop());
            }
            help.push(cur);
        }
        //����stack��
        while(!help.isEmpty()){
            stack.push(help.pop());
        }

    }


}
