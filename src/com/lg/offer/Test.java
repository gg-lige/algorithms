package com.lg.offer;

import com.lg.sort.MaxPQ;

import java.util.*;

/**
 * Created by lg on 2018/3/30.
 */
public class Test {
    public static void main(String[] args) {
//        char[] matrix = {'a', 'b', 'c', 'e', 's', 'f', 'c', 's'};
//        char[] a = {'b', 'c', 'c', 's'};
//        hasPath(matrix, 2, 4, a);
        int[] value={
                9,11,8,5,7,12,16,14
        };
        System.out.println(value);

    }


    public static  int longestSubstringWithDuplication(String s) {
        int curLength =0;
        int maxLength =0;
        char[] str = s.toCharArray();
        int[] position = new int[26];
        for (int i = 0; i < str.length; i++) {
            int prevIndex = position[str[i]-'a'];
            if(prevIndex <0 || i - prevIndex > curLength)
                curLength++;
            else {
                if(curLength>maxLength)
                    maxLength = curLength;
                curLength=i -prevIndex;
            }
            position[str[i]-'a']=i;
        }
        if(curLength>maxLength)
            maxLength= curLength;
        return maxLength;
    }
 //   public static int longestSubstringWith

    //之字形打印
    public ArrayList<ArrayList<Integer>> Print3(TreeNode pRoot) {
        ArrayList<ArrayList<Integer>> result = new ArrayList<ArrayList<Integer>>();
        if (pRoot == null)
            return result;

        Stack<TreeNode>[] levels = new Stack[2];
        ArrayList<Integer> layer = new ArrayList<Integer>();
        int current = 0;
        int next = 1;
        levels[current].push(pRoot);
        while (!levels[0].empty() || !levels[1].empty()) {
            TreeNode curNode = levels[0].peek();
            levels[0].pop();
            layer.add(curNode.val);

            if (current == 0) {
                if (curNode.left != null)
                    levels[next].push(curNode.left);
                if (curNode.right != null)
                    levels[next].push(curNode.right);
            } else {
                if (curNode.right != null)
                    levels[next].push(curNode.right);
                if (curNode.left != null)
                    levels[next].push(curNode.left);
            }

            if (levels[current].empty()) {
                result.add(layer);
                current = 1 - current;
                next = 1 - next;
                layer = new ArrayList<Integer>();
            }
        }
        return result;
    }

    //65
    public ArrayList<Integer> maxInWindows(int[] num, int size) {
        ArrayList<Integer> maxInWindows = new ArrayList<>();
        if (num.length >= size && size >= 1) {
            Deque<Integer> index = new LinkedList<Integer>();
            for (int i = 0; i < size; i++) {
                while (!index.isEmpty() && num[i] >= num[index.getLast()])
                    index.removeLast();
                index.offerLast(i);
            }
            for (int i = size; i < num.length; ++i) {
                maxInWindows.add(num[index.getFirst()]);
                while (!index.isEmpty() && num[i] >= num[index.getLast()])
                    index.removeLast();
                if (!index.isEmpty() & index.getFirst() <= (int) (i - size))
                    index.removeFirst();
                index.offerLast(i);
            }
            maxInWindows.add(num[index.getFirst()]);
        }
        return maxInWindows;
    }

    //66
    public static boolean hasPath(char[] matrix, int rows, int cols, char[] str) {
        if (matrix == null || rows < 1 || cols < 1 || str == null)
            return false;
        boolean[] visited = new boolean[rows * cols];
        int pathLength = 0;
        for (int row = 0; row < rows; ++row) {
            for (int col = 0; col < cols; ++col) {
                if (hasPathCore(matrix, rows, cols, row, col, str, pathLength, visited))
                    return true;
            }
        }
        return false;
    }

    public static boolean hasPathCore(char[] matrix, int rows, int cols, int row, int col, char[] str, int pathLength, boolean[] visited) {
        if (str[pathLength] == '\0')
            return true;

        boolean hasPath = false;
        if (row >= 0 && row <= rows && col >= 0 && col <= cols && matrix[row * cols + col] == str[pathLength] && !visited[row * cols + col]) {
            pathLength++;
            visited[row * cols + col] = true;
            hasPath = hasPathCore(matrix, rows, cols, row, col - 1, str, pathLength, visited)
                    || hasPathCore(matrix, rows, cols, row - 1, col, str, pathLength, visited)
                    || hasPathCore(matrix, rows, cols, row, col + 1, str, pathLength, visited)
                    || hasPathCore(matrix, rows, cols, row + 1, col, str, pathLength, visited);
            if (hasPath) {
                --pathLength;
                visited[row * cols + col] = false;
            }
        }
        return hasPath;
    }


    //67.
    public int movingCount(int threshold, int rows, int cols) {
        boolean[] visited = new boolean[rows * cols];
        for (int i = 0; i < rows * cols; ++i)
            visited[i] = false;

        int count = movingCountCore(threshold, rows, cols, 0, 0, visited);
        return count;
    }

    public int movingCountCore(int threshold, int rows, int cols, int row, int col, boolean[] visited) {
        int count = 0;
        if (check(threshold, rows, cols, row, col, visited)) {
            visited[row * cols + col] = true;
            count = 1 + movingCountCore(threshold, rows, cols, row - 1, col, visited)
                    + movingCountCore(threshold, rows, cols, row, col - 1, visited)
                    + movingCountCore(threshold, rows, cols, row + 1, col, visited)
                    + movingCountCore(threshold, rows, cols, row, col + 1, visited);
        }
        return count;

    }

    public boolean check(int threshold, int rows, int cols, int row, int col, boolean[] visited) {
        if (row >= 0 && row < rows && col >= 0 && col <= cols
                && getDigitSum(row) + getDigitSum(col) <= threshold && !visited[row * cols + col])
            return true;
        return false;
    }

    public int getDigitSum(int number) {
        int sum = 0;
        while (number > 0) {
            sum += number % 10;
            number /= 10;
        }
        return sum;
    }
}
