
package com.lg.offer;

import java.util.*;


public class Offer {
    // 查找二叉树的最长路径
    public int findMaxPath(BinaryTreeNode t) {
        if (t != null) {
            int lpath = findMaxPath(t.left);
            int rpath = findMaxPath(t.right);
            t.floor = Math.max((t.left != null) ? t.left.floor : 0, (t.right != null) ? t.right.floor : 0) + 1;

            return Math.max(Math.max(lpath, rpath),
                    ((t.left != null) ? t.left.floor : 0) + ((t.right != null) ? t.right.floor : 0));
        }
        return 0;
    }

    //最长不含重复字符的子字符串的长度
    public static int longestSubstringWithDuplication(String s) {
        int curLength = 0;
        int maxLength = 0;
        char[] str = s.toCharArray();
        int[] position = new int[26];
        for (int i = 0; i < str.length; i++) {
            int prevIndex = position[str[i] - 'a'];
            if (prevIndex < 0 || i - prevIndex > curLength)
                curLength++;
            else {
                if (curLength > maxLength)
                    maxLength = curLength;
                curLength = i - prevIndex;
            }
            position[str[i] - 'a'] = i;
        }
        if (curLength > maxLength)
            maxLength = curLength;
        return maxLength;
    }

    // 两字符串的最长公共字串
    public int findLongestSubstring(String str1, String str2) {
        int len1 = str1.length();
        int len2 = str2.length();
        int result = 0; // 记录最长公共子串长度
        int c[][] = new int[len1 + 1][len2 + 1];
        for (int i = 0; i <= len1; i++) {
            for (int j = 0; j <= len2; j++) {
                if (i == 0 || j == 0) {
                    c[i][j] = 0;
                } else if (str1.charAt(i - 1) == str2.charAt(j - 1)) {
                    c[i][j] = c[i - 1][j - 1] + 1;
                    result = Math.max(c[i][j], result);
                } else {
                    c[i][j] = 0;
                }
            }
        }
        return result;
    }

    //用一个栈来排序另一个栈
    private static void sortStackByStack(Stack<Integer> stack) {
        Stack<Integer> help = new Stack<>();
        while (!stack.isEmpty()) {
            int cur = stack.pop();
            while (!help.isEmpty() && cur > help.peek()) {
                stack.push(help.pop());
            }
            help.push(cur);
        }
        //倒回stack中
        while (!help.isEmpty()) {
            stack.push(help.pop());
        }

    }

    // 3.二维数组中的查找
    public boolean Find(int target, int[][] array) {
        int rows = array.length;
        int cols = array[0].length;
        boolean find = false;
        if (rows > 0 && cols > 0 && array != null) {
            int row = 0;
            int col = cols - 1;
            while (row < rows && col >= 0) {
                if (array[row][col] == target) {
                    find = true;
                    break;
                } else if (array[row][col] > target) {
                    --col;
                } else {
                    ++row;
                }
            }
        }
        return find;
    }

    // 4.替换空格
    public String replaceSpace(StringBuffer str) {

        int length = str.length();
        int spaceCount = 0, i;
        for (i = 0; i < length; i++) {
            if (str.charAt(i) == ' ')
                spaceCount++;
        }

        int newlength = length + 2 * spaceCount;
        int indexOld = length - 1;
        int indexNew = newlength - 1;
        str.setLength(newlength);

        for (; indexOld >= 0; indexOld--) {
            if (str.charAt(indexOld) == ' ') {
                str.setCharAt(indexNew--, '0');
                str.setCharAt(indexNew--, '2');
                str.setCharAt(indexNew--, '%');
            } else {
                str.setCharAt(indexNew--, str.charAt(indexOld));
            }
        }

        return str.toString();

    }

    // 5.从尾到头打印链表
    public ArrayList<Integer> printListFromTailToHead(ListNode listNode) {
        ArrayList<Integer> outputNode = new ArrayList<Integer>();
        ListNode nextNode = listNode;
        // 使用递归来实现
        if (nextNode != null) {
            if (nextNode.next != null) {
                outputNode = printListFromTailToHead(nextNode.next);
            }
            outputNode.add(nextNode.val);
        }
        return outputNode;
    }

    // 6.重建二叉树(根据前序和中序序列)
    public TreeNode reConstructBinaryTree(int[] pre, int[] in) {
        if (pre == null || in == null)
            return null;
        else
            return reConstructBinaryTree(pre, 0, pre.length - 1, in, 0, in.length - 1);
    }

    // 递归
    private TreeNode reConstructBinaryTree(int[] pre, int sPre, int ePre, int[] in, int sIn, int eIn) {

        if (sPre > ePre || sIn > eIn)
            return null;
        TreeNode root = new TreeNode(pre[sPre]); // 前序寻找根节点
        // 中序中找到根节点，前面的为左子树，后面的为柚子树
        for (int i = sIn; i <= eIn; i++) {
            if (in[i] == pre[sPre]) {
                root.left = reConstructBinaryTree(pre, sPre + 1, sPre + i - sIn, in, sIn, i - 1);
                root.right = reConstructBinaryTree(pre, sPre + i - sIn + 1, ePre, in, i + 1, eIn);
                break;
            }

        }
        return root;
    }

    // 7.用两个栈实现队列
    Stack<Integer> stack1 = new Stack<Integer>();
    Stack<Integer> stack2 = new Stack<Integer>();

    // 入队
    public void push(int node) {
        stack1.push(node);
    }

    public int pop() {
        if (stack2.empty()) {
            while (!stack1.empty()) {
                int temp = stack1.pop();
                stack2.push(temp);
            }
        }
        if (stack2.empty())
            System.out.println("queue is empty");

        return stack2.pop();

    }

    // 8.旋转数组的最小数字
    public int minNumberInRotateArray(int[] array) {
        if (array == null || array.length == 0)
            return 0;
        // 旋转数组主要查找递增数组突然出现一个下降的数字时，下降的数字即为所查。
        int index1 = 0, index2 = array.length - 1;
        int mid = -1;

        while (array[index1] >= array[index2]) {
            if (index2 - index1 == 1) {
                mid = index2;
                break;
            }
            mid = (index1 + index2) / 2;

            // 如果下标1、下标2、中间标的指向数字相同，则需要顺序查找。
            if (array[index1] == array[index2] && array[index1] == array[mid]) { // ??短路操作
                return minInOrder(array, index1, index2);
            }
            if (array[mid] >= array[index1])
                index1 = mid;
            if (array[mid] <= array[index2])
                index2 = mid;

        }
        return array[mid];
    }

    private int minInOrder(int[] array, int index1, int index2) {
        int result = array[index1];
        for (int i = index1 + 1; i <= index2; i++) {
            if (array[i] < result)
                result = array[i];

        }
        return result;
    }

    // 9.斐波那契数列
    public int Fibonacci(int n) {
        // 用循环方式来求解时间复杂度会好些，而递归方式的由于每次函数调用过程都会使用栈来存储函数变量、参数、返回地址等
        int[] result = {0, 1};
        if (n < 0)
            System.out.println("范围不正确");
        else if (n < 2) {
            return result[n];
        }

        int first = 0;
        int second = 1;
        int fib = 0;
        for (int i = 2; i <= n; i++) {
            fib = first + second;
            first = second;
            second = fib;
        }
        return fib;
    }

    // 9.跳台阶，矩形覆盖
    public int JumpFloor(int target) {
        int[] result = {1, 2};
        if (target < 1)
            System.out.println("台阶输入不正确");
        else if (target < 3) {
            return result[target - 1];
        }

        int first = 1;
        int second = 2;
        int floor = 0;
        for (int i = 3; i <= target; i++) {
            floor = first + second;
            first = second;
            second = floor;

        }
        return floor;
    }

    // 9.变态跳台阶
    public int JumpFloorII(int target) {
        if (target < 1)
            System.out.println("台阶输入不正确");
        int floor = 1;
        for (int i = 2; i <= target; i++) {
            floor = 2 * floor;

        }
        return floor;

    }

    // 10.二进制中1的个数
    public int NumberOf1(int n) {
        // 将该整数-1 并与原整数进行与操作，会将原整数最右边的1变为0.有多少个1，则进行多少次操作。
        int count = 0;
        while (n != 0) {
            n = (n - 1) & n;
            count++;
        }
        return count;
    }

    // 11.数值的整数次方
    boolean invalidInput = false;// 全局变量

    public double Power(double base, int exponent) {
        invalidInput = false;
        if (exponent < 0 && equal(base, 0.0)) {
            invalidInput = true;
            return 0.0;
        }
        int absExponent = exponent;
        if (exponent < 0) {
            absExponent = -exponent;
        }
        double result = PowerWithUnsignedExponent(base, absExponent);
        if (exponent < 0) {
            result = 1 / result;
        }
        return result;
    }

    double PowerWithUnsignedExponent(double base, int exponent) {
        // 递归方式实现
        if (exponent == 0)
            return 1;
        if (exponent == 1)
            return base;

        double result = PowerWithUnsignedExponent(base, exponent >> 1);// 右移表示
        // 除以2
        result *= result;
        if ((exponent & 0x1) == 1) { // 与1与判奇偶
            result *= base;
        }
        return result;
    }

    boolean equal(double n1, double n2) {
        if ((n1 - n2 > -0.0000001) && (n1 - n2) < 0.0000001)
            return true;
        else
            return false;
    }

    // 14.调整数组顺序使奇数位于偶数前面
    public void reOrderArray(int[] a) {
        if (a == null || a.length == 0)
            return;
        int i = 0, j;
        while (i < a.length) {
            while (i < a.length && !isEven(a[i])) // i从左向右找到第一个偶数
                i++;
            j = i + 1;// j从下一个开始搜索第一个奇数
            while (j < a.length && isEven(a[j]))
                j++;
            if (j < a.length) { // 找到奇数后，将所有偶数后移，并在前面插入奇数
                int tmp = a[j];
                for (int j2 = j - 1; j2 >= i; j2--) {
                    a[j2 + 1] = a[j2];
                }
                a[i++] = tmp;
            } else {
                break;
            }
        }
    }

    boolean isEven(int n) {
        if (n % 2 == 0)
            return true;
        return false;
    }

    // 15.链表中倒数第k个结点
    public ListNode FindKthToTail(ListNode head, int k) {
        if (head == null || k == 0)
            return null;

        ListNode p1 = head;
        ListNode p2 = head;

        for (int i = 0; i < k - 1; i++) {
            if (p1.next != null)
                p1 = p1.next;
            else
                return null;
        }
        while (p1.next != null) {
            p1 = p1.next;
            p2 = p2.next;
        }
        return p2;
    }

    // 16.反转链表（循环）
    public ListNode ReverseList_recur(ListNode head) {
        if (head == null)
            return null;
        ListNode pReverseNode = null;
        ListNode node = head;
        ListNode preNode = null;
        while (node != null) {
            ListNode nextNode = node.next;
            if (nextNode == null) {
                pReverseNode = node;
            }
            node.next = preNode;
            preNode = node;
            node = nextNode;
        }
        return pReverseNode;
    }

    public ListNode ReverseList_digui(ListNode head) {
        if (head == null || head.next == null)
            return head;
        ListNode pReverseNode = ReverseList_digui(head.next);
        head.next.next = head;
        head.next = null;
        return pReverseNode;
    }

    // 17.合并两个排序的链表
    public ListNode Merge(ListNode list1, ListNode list2) {
        if (list1 == null)
            return list2;
        if (list2 == null)
            return list1;

        if (list1.val < list2.val) {
            list1.next = Merge(list1.next, list2);
            return list1;
        } else {
            list2.next = Merge(list1, list2.next);
            return list2;
        }

    }

    // 18.树的子结构
    public boolean HasSubtree(TreeNode root1, TreeNode root2) {
        boolean result = false;
        if (root1 != null && root2 != null) {
            if (root1.val == root2.val) {
                result = ifHasSubtree(root1, root2);
            }
            if (!result) {
                result = HasSubtree(root1.left, root2);
            }
            if (!result) {
                result = HasSubtree(root1.right, root2);
            }
        }
        return result;
    }

    private boolean ifHasSubtree(TreeNode root1, TreeNode root2) {
        if (root2 == null)
            return true;// 注意先判断这个。表明将B的每个节点都判断过了
        if (root1 == null)
            return false;

        if (root1.val != root2.val)
            return false;

        return ifHasSubtree(root1.left, root2.left) && ifHasSubtree(root1.right, root2.right);

    }

    // 19.二叉树的镜像
    public void Mirror(TreeNode root) {
        if (root == null)
            return;
        if (root.left == null && root.right == null)
            return;
        TreeNode temp = root.left;
        root.left = root.right;
        root.right = temp;

        if (root.left != null)
            Mirror(root.left);
        if (root.right != null)
            Mirror(root.right);

    }

    // 20.顺时针打印矩阵
    ArrayList<Integer> result = new ArrayList<Integer>();

    public ArrayList<Integer> printMatrix(int[][] matrix) {
        int rows = matrix.length;
        int cols = matrix[0].length;
        if (matrix == null || rows <= 0 || cols <= 0)
            return null;

        int start = 0; // 左上角开始坐标，可以理解为圈数
        while (start * 2 < rows && start * 2 < cols) {
            result = printInCircle(matrix, cols, rows, start);
            start++;
        }
        return result;
    }

    private ArrayList<Integer> printInCircle(int[][] matrix, int col, int row, int start) {
        int endX = col - start - 1;
        int endY = row - start - 1;
        // 打印第一行，从左到右横着打
        for (int i = start; i <= endX; i++)
            result.add(matrix[start][i]);

        // 从上到下打印一列
        if (start < endY) {
            for (int i = start + 1; i <= endY; i++)
                result.add(matrix[i][endX]);
        }

        // 从右到左打印一行
        if (start < endX && start < endY) {
            for (int i = endX - 1; i >= start; i--)
                result.add(matrix[endY][i]);
        }

        // 从下到上打印一列
        if (start < endX && start < endY - 1) {
            for (int i = endY - 1; i >= start + 1; i--)
                result.add(matrix[i][start]);
        }
        return result;
    }

    // 21.包含min函数的栈
    private Stack<Integer> data = new Stack<Integer>();
    private Stack<Integer> ass = new Stack<Integer>();

    public void push2(int node) {
        data.push(node);
        if (ass.empty() || ass.peek() > node)
            ass.push(node); // 辅助栈中存每次操作的最小值
        else
            ass.push(ass.peek());
    }

    public void pop2() {
        if (data.empty() || ass.empty())
            return;
        data.pop();
        ass.pop();
    }

    public int top() {
        if (data.empty())
            return (Integer) null;
        else
            return data.peek();
    }

    public int min() {
        if (ass.empty())
            return (Integer) null;
        else
            return ass.peek();
    }

    // 22.栈的压入、弹出序列
    public boolean IsPopOrder(int[] pushA, int[] popA) {
        if (pushA.length == 0 || popA.length == 0)
            return false;
        Stack<Integer> s = new Stack<Integer>();
        // 用于标识弹出序列的位置
        int popIndex = 0;
        for (int i = 0; i < pushA.length; i++) {
            s.push(pushA[i]);// 如果栈不为空，且栈顶元素等于弹出序列
            while (!s.empty() && s.peek() == popA[popIndex]) {
                // 出栈
                s.pop();
                // 弹出序列向后一位
                popIndex++;
            }
        }
        return s.empty();
    }

    // 23.从上往下打印二叉树
    public ArrayList<Integer> PrintFromTopToBottom(TreeNode root) {
        ArrayList<Integer> result = new ArrayList<Integer>();
        if (root == null)
            return result;
        Queue<TreeNode> q = new LinkedList<TreeNode>();
        q.offer(root);
        while (!q.isEmpty()) {
            TreeNode treeNode = q.poll();
            if (treeNode.left != null) {
                q.offer(treeNode.left);
            }
            if (treeNode.right != null) {
                q.offer(treeNode.right);
            }
            result.add(treeNode.val);
        }
        return result;
    }

    // 24.某序列是否为二叉搜索树的后序遍历序列
    public boolean VerifySquenceOfBST(int[] sequence) {
        if (sequence.length <= 0 || sequence == null)
            return false;

        return verifySequenceOfBST(sequence, 0, sequence.length - 1);
    }

    private boolean verifySequenceOfBST(int[] sequence, int start, int end) {
        if (start >= end)
            return true;
        int index = start;
        while (index < end - 1 && sequence[index] < sequence[end]) {
            index++;
        }
        int right = index;
        while (index < end - 1 && sequence[index] > sequence[end]) {
            index++;
        }
        if (index != end - 1) {
            return false;
        }
        index = right;
        return verifySequenceOfBST(sequence, start, index - 1) && verifySequenceOfBST(sequence, index, end - 1);

    }

    //25.二叉树中和为某一值的路径
    private ArrayList<ArrayList<Integer>> paths = new ArrayList<ArrayList<Integer>>();
    private ArrayList<Integer> p = new ArrayList<>();

    public ArrayList<ArrayList<Integer>> FindPath(TreeNode root, int target) {
        if (root == null)
            return paths;
        p.add(root.val);
        target -= root.val; //使用target 减
        if (target == 0 && root.left == null && root.right == null)
            paths.add(new ArrayList<Integer>(p));

        FindPath(root.left, target);
        FindPath(root.right, target);
        p.remove(p.size() - 1); ///????
        return paths;
    }

    //26.复杂链表的复制
    public RandomListNode Clone(RandomListNode pHead) {
        if (pHead == null) {
            return null;
        }

        RandomListNode pCur = pHead;
        //复制链表a,b,c    a,a',b,b',c,c'
        while (pCur != null) {
            RandomListNode clone = new RandomListNode(pCur.label);
            clone.next = pCur.next;
            pCur.next = clone;
            pCur = clone.next;
        }

        //复制兄弟节点
        pCur = pHead;
        while (pCur != null) {
            if (pCur.random != null) {
                pCur.next.random = pCur.random.next;
            }
            pCur = pCur.next.next;
        }

        //拆分
        pCur = pHead;
        RandomListNode head = pHead.next;
        RandomListNode cur = pHead.next;

        while (pCur != null) {
            pCur.next = pCur.next.next;
            if (cur.next != null) {
                cur.next = cur.next.next;
            }
            pCur = pCur.next;
            cur = cur.next;

        }
        return head;
    }

    //27.二叉搜索树转换为双向链表
    TreeNode head = null;
    TreeNode realHead = null;

    public TreeNode Convert(TreeNode pRootOfTree) {
        if (pRootOfTree == null)
            return null;
        Convert(pRootOfTree.left);
        if (head == null) { //递归左树
            head = pRootOfTree;
            realHead = pRootOfTree;
        } else {
            head.right = pRootOfTree;
            pRootOfTree.left = head;
            head = pRootOfTree;
        }
        Convert(pRootOfTree.right);
        return realHead;
    }

    //28.打印字符串的所有排列
    public ArrayList<String> Permutation(String str) {
        ArrayList<String> result = new ArrayList<String>();
        if (str == null || str.length() == 0)
            return result;

        int start = 0;
        reArrange(str.toCharArray(), start, result);
        Collections.sort(result);
        return (ArrayList) result;
    }

    public void reArrange(char[] str, int start, List<String> result) {
        if (start == str.length - 1) {
            String s = String.valueOf(str);//String的静态方法，得到具体的值
            if (!result.contains(s))
                result.add(s);
        } else {
            for (int i = start; i < str.length; i++) {//abc
                swap(str, start, i);// 交换数组第一个元素与后续的元素bac
                reArrange(str, start + 1, result);// 后续元素递归全排列
                swap(str, start, i); // 将交换后的数组还原  //abc
            }
        }
    }

    public void swap(char[] s, int i, int j) {
        char temp = s[i];
        s[i] = s[j];
        s[j] = temp;
    }

    //29.数组中出现次数超过一半的数字（快排思想，会修改数组）
    public int MoreThanHalfNum_Solution(int[] array) {
        int length = array.length;
        if (length < 0)
            return 0;

        int result = array[0];
        int times = 1;
        for (int i = 1; i < length; i++) {
            if (times == 0) {
                result = array[i];
                times = 1;
            } else if (array[i] == result)
                times++;
            else
                times--;
        }

        times = 0;//判断最大times是否超过数组长的一半
        for (int i = 0; i < length; i++) {
            if (array[i] == result)
                times++;
        }
        if (times * 2 <= length) {
            System.out.println(times);
            return 0;
        } else
            return result;

    }

    // 法 2  每次约减不同的两个数、最后剩下的一个（数组特性，不会修改数组）
    public int MoreThanHalfNum_Solution2(int[] array) {
        int length = array.length;
        if (array == null || length == 0)
            return 0;
        int lo = 0;
        int hi = length - 1;
        int middle = length >> 1;
        if (lo >= hi)
            return array[0];
        int j = partition(array, lo, hi);
        while (j != middle) {
            if (middle < j) {  //中位数在j的左边
                hi = j - 1;
                j = partition(array, lo, hi);
            } else {
                lo = j + 1;
                j = partition(array, lo, hi);
            }
        }
        int result = array[middle];
        int times = 0; //判断最大times是否超过数组长的一半
        for (int i = 0; i < length; ++i) {
            if (array[i] == result)
                times++;
        }
        if (times * 2 <= length) {
            System.out.println("0");
            return 0;
        } else
            return result;

    }

    public int partition(int[] a, int lo, int hi) {
        if (lo == hi)  //注意当lo==最后一个元素时，下面的a[++i]会出错
            return lo;
        int i = lo, j = hi + 1;
        int v = a[lo];
        while (true) {
            while (a[++i] < v) if (i == hi) break;
            while (v < a[--j]) if (j == lo) break;
            if (i >= j) break;
            exch(a, i, j);
        }
        exch(a, lo, j);
        return j;
    }

    public void exch(int[] a, int i, int j) {
        int t = a[i];
        a[i] = a[j];
        a[j] = t;
    }


    //30. 最小的K个数(法1)
    public ArrayList<Integer> GetLeastNumbers_Solution(int[] input, int k) {
        ArrayList<Integer> result = new ArrayList<Integer>();
        int length = input.length;
        if (length == 0 || input == null || k <= 0 || length < k)
            return result;

        PriorityQueue<Integer> queue = new PriorityQueue<Integer>(k, new Comparator<Integer>() {
            @Override
            public int compare(Integer n1, Integer n2) {
                return n2.compareTo(n1); //排成大顶堆,注意return
            }
        });

        for (int i = 0; i < length; i++) {
            if (queue.size() < k)
                queue.offer(input[i]);
            else if (queue.peek() > input[i]) {
                queue.poll();
                queue.offer(input[i]);
            }
        }
        for (Integer e : queue) {
            result.add(e);
        }
        return result;
    }

    //法二(快排数组格式)
    public ArrayList<Integer> GetLeastNumbers_Solution2(int[] input, int k) {
        ArrayList<Integer> result = new ArrayList<Integer>();
        int length = input.length;
        if (input == null || k > length || length <= 0 || k <= 0)
            return result;

        int lo = 0, hi = length - 1;
        int j = partition(input, lo, hi);
        while (j != k - 1) {
            if (k - 1 < j) {
                hi = j - 1;
                j = partition(input, lo, hi);
            } else {
                lo = j + 1;
                j = partition(input, lo, hi);
            }
        }
        for (int i = 0; i < k; i++) {
            result.add(input[i]);
        }
        return result;
    }


    //31.连续子数组的最大和
    boolean invalidinput = false; //无效输入

    public int FindGreatestSumOfSubArray(int[] array) {
        if (array == null || array.length <= 0) {//total记录累计值，maxSum记录和最大
            invalidinput = true;
            return 0;
        }
        int length = array.length;
        int curSum = 0;
        int max = array[0];
        for (int i = 0; i < length; i++) {
            if (curSum <= 0)
                curSum = array[i];
            else
                curSum += array[i];
            if (curSum > max)
                max = curSum;
        }
        return max;
    }

    //32.从1到n整数中1出现的次数
    public int NumberOf1Between1AndN_Solution(int n) {
        int count = 0;
        StringBuilder s = new StringBuilder();
        for (int i = 1; i < n + 1; i++) {
            s.append(i);
        }
        String str = s.toString();
        for (int i = 0; i < str.length(); i++) {
            if (str.charAt(i) == '1')
                count++;
        }
        return count;
    }

    //33.把数组中的数按某顺序排成最小的数
    public String PrintMinNumber(int[] numbers) {
        int length = numbers.length;
        ArrayList<Integer> list = new ArrayList<>();
        for (int i = 0; i < length; i++)
            list.add(numbers[i]);

        Collections.sort(list, new Comparator<Integer>() {//注意Integer不能省
            public int compare(Integer e1, Integer e2) {
                String a = e1 + "" + e2;//“”加到中间，否则会先计算
                String b = e2 + "" + e1;
                return a.compareTo(b);
            }
        });
        String result = "";
        for (int i : list)
            result += i;
        return result;

    }

    //34.第n个丑数
    public int GetUglyNumber_Solution(int index) {
        if (index <= 0)
            return 0;

        int[] ugly = new int[index];
        ugly[0] = 1;
        int nextUglyIndex = 1;
        int t2 = 0, t3 = 0, t5 = 0;
        for (int i = 1; i < index; i++) {
            ugly[i] = Math.min(Math.min(ugly[t2] * 2, ugly[t3] * 3), ugly[t5] * 5);
            if (ugly[t2] * 2 == ugly[i]) t2++;
            if (ugly[t3] * 3 == ugly[i]) t3++;
            if (ugly[t5] * 5 == ugly[i]) t5++;
        }
        return ugly[index - 1];
    }

    //35.第一个只出现一次的字符的位置(模拟数组为hashtable)
    public int FirstNotRepeatingChar1(String str) {
        if (str == null)
            return 0;

        int[] hashTable = new int[256];
        for (int i = 0; i < 256; i++)
            hashTable[i] = 0;

        char[] strArray = str.toCharArray();
        for (char c : strArray)
            hashTable[(int) c]++;

        for (int i = 0; i < strArray.length; i++) {
            if (hashTable[strArray[i]] == 1) return i;
        }
        return -1;

    }

    //法二（直接使用hashMap）
    public int FirstNotRepeatingChar2(String str) {
        LinkedHashMap<Character, Integer> map = new LinkedHashMap<Character, Integer>();
        for (int i = 0; i < str.length(); i++) {
            if (map.containsKey(str.charAt(i))) {
                int time = map.get(str.charAt(i));
                map.put(str.charAt(i), ++time);
            } else {
                map.put(str.charAt(i), 1);
            }
        }
        int pos = -1;
        for (int i = 0; i < str.length(); i++) {
            char c = str.charAt(i);
            if (map.get(c) == 1) {
                return i;
            }
        }
        return pos;
    }

    //36. 数组中的逆序对个数 %1000000007
    public int InversePairs(int[] array) {
        int length = array.length;
        if (array == null || length < 0)
            return 0;

        int[] aux = new int[length];
        for (int i = 0; i < length; i++)
            aux[i] = array[i];

        return InversePairsCore(array, aux, 0, length - 1);
    }

    public int InversePairsCore(int[] array, int[] aux, int start, int end) {
        if (start == end) {
            aux[start] = array[start];
            return 0;
        }
        int mid = start + (end - start) / 2;
        int left = InversePairsCore(array, aux, start, mid) % 1000000007;
        int right = InversePairsCore(array, aux, mid + 1, end) % 1000000007;

        int i = mid;
        int j = end;
        int auxIndex = end;
        int count = 0;
        while (i >= start && j >= mid + 1) {
            if (array[i] > array[j]) {
                aux[auxIndex--] = array[i--];
                count += j - mid;
                if (count >= 1000000007)//数值过大求余
                {
                    count %= 1000000007;
                }
            } else {
                aux[auxIndex--] = array[j--];
            }
        }

        for (; i >= start; --i)
            aux[auxIndex--] = array[i];
        for (; j >= mid + 1; --j)
            aux[auxIndex--] = array[j];
        for (int s = start; s <= end; s++) { //拷贝回原数组
            array[s] = aux[s];
        }
        return (left + right + count) % 1000000007;
    }


    // 37. 两个链表中的第一个公共子节点
    public ListNode FindFirstCommonNode(ListNode pHead1, ListNode pHead2) {
        if (pHead1 == null || pHead2 == null)
            return null;

        int length1 = getLength(pHead1);
        int length2 = getLength(pHead2);
        int len = Math.abs(length1 - length2);

        ListNode longLink = pHead1;
        ListNode shortLink = pHead2;
        if (length1 < length2) {
            longLink = pHead2;
            shortLink = pHead1;
        }

        for (int i = 0; i < len; i++) {
            longLink = longLink.next;
        }

        while (longLink != null && shortLink != null && longLink.val != shortLink.val) {
            longLink = longLink.next;
            shortLink = shortLink.next;

        }
        return longLink;
    }

    public static int getLength(ListNode node) {
        int n = 1;
        while (node.next != null) {
            n++;
            node = node.next;
        }
        return n;
    }

    //38.数字在排序数组中出现的次数（二分查找法）
    public int GetNumberOfK(int[] array, int k) {
        int length = array.length;
        int number = 0;
        if (array != null || length > 0) {
            int firstK = getFirstK(array, k, 0, length - 1);
            int lastK = getLastK(array, k, 0, length - 1);
            if (firstK != -1 && lastK != -1) {
                number = lastK - firstK + 1;
            }
        }
        return number;
    }

    private int getFirstK(int[] array, int k, int start, int end) {
        if (start > end) {
            return -1;
        }
        int mid = (start + end) >> 1;
        int middleData = array[mid];
        if (middleData == k) {
            if ((mid > 0 && array[mid - 1] != k) || mid == 0)
                return mid;
            else
                end = mid - 1;
        } else if (middleData > k)
            end = mid - 1;
        else
            start = mid + 1;
        return getFirstK(array, k, start, end);
    }

    private int getLastK(int[] array, int k, int start, int end) {
        if (start > end) {
            return -1;
        }
        int mid = (start + end) >> 1;
        int middleData = array[mid];
        if (middleData == k) {
            if ((mid < end && array[mid + 1] != k) || mid == end)
                return mid;
            else
                start = mid + 1;
        } else if (middleData < k)
            start = mid + 1;
        else
            end = mid - 1;
        return getLastK(array, k, start, end);
    }

    //39. 二叉树的深度
    public int TreeDepth(TreeNode pRoot) {
        if (pRoot == null) {
            return 0;
        }
        int left = TreeDepth(pRoot.left);
        int right = TreeDepth(pRoot.right);
        return Math.max(left, right) + 1;
    }

    //39.二叉树是否是平衡二叉树
    private boolean isBalanced = true; //后续遍历时，遍历到一个节点，其左右子树已经遍历  依次自底向上判断，每个节点只需要遍历一次

    public boolean IsBalanced_Solution(TreeNode root) {
        getDepth(root);
        return isBalanced;
    }

    public int getDepth(TreeNode root) {
        if (root == null)
            return 0;
        int left = getDepth(root.left);
        int right = getDepth(root.right);
        if (Math.abs(left - right) > 1)
            isBalanced = false;
        return Math.max(left, right) + 1;
    }

    //40.数组中只出现一次的数字（找出这两个只出现一次的数字）
    public void FindNumsAppearOnce(int[] array, int num1[], int num2[]) {
        int length = array.length;
        if (array == null || length < 2)
            return;

        int resultExclusiveOr = 0;
        for (int i = 0; i < length; ++i) {
            resultExclusiveOr ^= array[i]; //所有元素逐个做异或操作
        }

        int indexOf1 = FindFirstBitIs1(resultExclusiveOr); //找到做完异或操作的1的下标的位置

        for (int j = 0; j < length; j++) {
            if (IsBit1(array[j], indexOf1))  //将数组分为两部分，一部分为下标为1的，一部分为下标为0的
                num1[0] ^= array[j];
            else
                num2[0] ^= array[j];
        }
    }

    private int FindFirstBitIs1(int num) {
        int indexBit = 0;
        while ((num & 1) == 0 && indexBit < 8 * 4) {//找第一个1的下标,注意java 中没有sizeof
            num = num >> 1; //找到num最右边时1的位置
            indexBit++;
        }
        return indexBit;
    }


    boolean IsBit1(int num, int index) {
        num = num >> index;
        return (num & 1) == 1;
    }

    //41.递增序列中和为S的两个数字
    public ArrayList<Integer> FindNumbersWithSum(int[] array, int sum) {
        ArrayList<Integer> list = new ArrayList<>();
        int length = array.length;
        if (length < 1 || array == null)
            return list;

        int small = 0; //较小数字的下标
        int large = length - 1; //较大数字的下标
        while (small < large) {
            int cursum = array[small] + array[large];
            if (cursum == sum) {
                list.add(array[small]);
                list.add(array[large]);
                break;
            } else if (cursum > sum) {
                large--;
            } else
                small++;
        }
        return list;
    }

    //41. 和为S的连续正数序列  多组
    public ArrayList<ArrayList<Integer>> FindContinuousSequence(int sum) {
        ArrayList<ArrayList<Integer>> lists = new ArrayList<ArrayList<Integer>>();
        if (sum < 3)
            return lists; //和<3即为2时，没有递增序列

        int small = 1;
        int big = 2;
        int middle = (1 + sum) / 2;

        while (small < middle) {  //当small<(1+sum)/2的时候停止
            int cursum = sumOfList(small, big);
            if (cursum == sum) {  //
                ArrayList<Integer> l = new ArrayList<Integer>();
                for (int i = small; i <= big; i++)
                    l.add(i);
                lists.add(l);
                small++; //因为有多组满足的条件
            } else if (cursum < sum) { //small到big序列和小于sum，big++;
                big++;
            } else  //大于sum，small++;
                small++;
        }
        return lists;
    }

    int sumOfList(int s, int e) { //计算当前序列的和
        int sum = s;
        for (int i = s + 1; i <= e; i++)
            sum += i;
        return sum;
    }

    //42. 反转单词顺序列
    public String ReverseSentence(String str) {
        if (str.trim().equals("")) //去掉字符串首尾的空格
            return str;

        String[] s = str.split(" ");
        StringBuilder sb = new StringBuilder();
        for (int i = s.length; i > 0; i--) {
            sb.append(s[i - 1]);
            if (i > 1)
                sb.append(" ");//在每个单词之间加一个空格
        }
        return sb.toString();
    }

    //42.左旋转字符串（按几个字符）
    public String LeftRotateString(String str, int n) {
        char[] chars = str.toCharArray();
        int length = chars.length;
        if (length > 0 && n > 0 && n < length) {
            reverse(chars, 0, n - 1); //翻前n个字符
            reverse(chars, n, length - 1); // 翻后面length-n个字符
            reverse(chars, 0, length - 1);  //翻整个字符
            return String.valueOf(chars);
        }
        return str;
    }

    public void reverse(char[] chars, int s, int e) {
        char temp;
        while (s < e) {
            temp = chars[s];
            chars[s] = chars[e];
            chars[e] = temp;
            s++;
            e--;
        }
    }

    //43. n个骰子的点数和为s的概率 （递归）
    int gmaxValue = 6;

    public void PrintProbability(int n) {
        if (n < 1)
            return;
        int maxSum = n * gmaxValue;
        int[] probabilities = new int[maxSum - n + 1];
        for (int i = n; i <= maxSum; i++) //初始化
            probabilities[i - n] = 0;
        Probability(n, probabilities);
        int total = (int) Math.pow(gmaxValue, n);
        for (int i = n; i <= maxSum; i++) {
            double ratio = (double) probabilities[i - n] / total;
            System.out.printf("%d :%e\n", i, ratio);
        }
    }

    private void Probability(int n, int[] probabilities) {
        for (int i = 1; i <= gmaxValue; i++)
            Probability(n, n, i, probabilities);
    }

    private void Probability(int orig, int curr, int sum, int[] probabilities) {
        if (curr == 1)
            probabilities[sum - orig]++;
        else
            for (int i = 1; i <= gmaxValue; ++i)
                Probability(orig, curr - 1, i + sum, probabilities);
    }

    //44. 扑克牌顺子
    public boolean isContinuous(int[] numbers) {
        int numberOfZero = 0; //大小王
        int numberOfInterval = 0; //所有相邻数字间隔总数
        int length = numbers.length;

        if (length == 0)
            return false;
        //1.排序
        Arrays.sort(numbers);
        for (int i = 0; i < length - 1; i++) {
            //2.计算大小王个数
            if (numbers[i] == 0) {
                numberOfZero++;
                continue;
            }
            //3.对子直接返回
            if (numbers[i] == numbers[i + 1]) {
                return false;
            }
            numberOfInterval += numbers[i + 1] - numbers[i] - 1;
        }
        return numberOfZero >= numberOfInterval ? true : false;
    }

    //45. 圆圈中剩下的最后一个数
    public int LastRemaining_Solution(int n, int m) {
        if (m == 0 || n == 0) {
            return -1;
        }
        LinkedList<Integer> data = new LinkedList<>();
        for (int i = 0; i < n; i++) {
            data.add(i);
        }
        int index = -1;
        while (data.size() > 1) {
            index = (index + m) % data.size();
            data.remove(index);
            index--;
        }
        return data.get(0);
    }

    //47. 不用加减乘除做加法（位运算）
    public int Add(int num1, int num2) {
        while (num2 != 0) { //没有进位时停止
            int sum = num1 ^ num2; //1.异或不进位
            int jinwei = (num1 & num2) << 1; //2.进位
            num1 = sum;
            num2 = jinwei;
        }
        return num1;
    }

    //49.把字符串转换成整数（注意考虑正负、越界、非数字）
    public int StrToInt(String str) {
        if (str == null || str.length() == 0)
            return 0;
        char[] c = str.toCharArray();
        boolean minus = false;
        int num = 0;
        int i = 0;
        //数组溢出：下标大于数组长度！比如c.length ==1,当有c[1]出现时则数组溢出
        if (c[i] == '+')
            ++i;
        else if (c[i] == '-') {
            ++i;
            minus = true;
        }
        if (i < c.length)
            num = StrToIntCore(c, minus, i);//i表示第一个数字的下标
        return num;
    }

    public int StrToIntCore(char[] str, boolean minus, int i) {
        int num = 0;
        for (int j = i; j < str.length; j++) {
            if (str[j] >= '0' && str[j] <= '9') {
                int flag = minus ? -1 : 1;  //定义正负
                num = num * 10 + flag * (str[j] - '0');
                if ((!minus && num > Integer.MAX_VALUE) || (minus && num < Integer.MIN_VALUE))//处理越界
                {
                    num = 0;
                    break;
                }
            } else {
                num = 0;
                break;
            }
        }
        return num;
    }

    //50. 二树中两个节点的最低公共祖先
    public TreeNode getLastCommonParent(TreeNode root, TreeNode n1, TreeNode n2) {
        if (root == null || n1 == null || n2 == null)
            return null;
        ArrayList<TreeNode> p1 = new ArrayList<>();
        ArrayList<TreeNode> p2 = new ArrayList<>();
        getNodePath(root, n1, p1);
        getNodePath(root, n2, p2);

        return getLastCommonNode(p1, p2);
    }

    private boolean getNodePath(TreeNode root, TreeNode n1, ArrayList<TreeNode> path) {
        if (root == n1)
            return true;
        path.add(root);
        boolean found = false;
        if (!found && root.left != null) {
            found = getNodePath(root.left, n1, path);
        }
        if (!found && root.right != null) {
            found = getNodePath(root.right, n1, path);
        }
        if (!found) path.remove(path.get(path.size() - 1));
        return found;
    }

    private TreeNode getLastCommonNode(ArrayList<TreeNode> p1, ArrayList<TreeNode> p2) {
        TreeNode last = null;
        for (int i = 0; i < p1.size(); i++) {
            if (p1.get(i) != p2.get(i))
                break;
            last = p1.get(i);
        }
        return last;
    }

    //51. 数组中重复的数字
    public boolean duplicate(int numbers[], int length, int[] duplication) {
        if (numbers == null || length < 1)
            return false;
        for (int i = 0; i < length; i++)
            if (numbers[i] < 0 || numbers[i] > length - 1) //题目要求
                return false;

        for (int i = 0; i < length; i++) {
            while (numbers[i] != i) {
                //判是否有重复
                if (numbers[i] == numbers[numbers[i]]) {
                    duplication[0] = numbers[i]; //返回重复的数字
                    return true;
                }
                //不等于时要交换
                int temp = numbers[i];
                numbers[i] = numbers[temp];
                numbers[temp] = temp;
            }
        }
        return false;
    }

    //52.构建乘积数组 B[i]=A[0..i-1]*A[i+1..n-1]
    public int[] multiply(int[] A) {
        int length = A.length;
        int[] B = new int[length];
        if (length != 0) {
            //下三角连乘
            B[0] = 1;
            for (int i = 1; i < length; i++)
                B[i] = B[i - 1] * A[i - 1];
            //上三角连乘
            int temp = 1;
            for (int j = length - 2; j >= 0; j--) {
                temp = temp * A[j + 1];
                B[j] = temp * B[j];
            }
        }
        return B;
    }

    //53.正则表达时的匹配
    public boolean match(char[] str, char[] pattern) {
        if (str == null || pattern == null)
            return false;

        int strIndex = 0;
        int patternIndex = 0;
        return matchCore(str, strIndex, pattern, patternIndex);
    }

    public boolean matchCore(char[] str, int strIndex, char[] pattern, int patternIndex) {
        if (strIndex == str.length && patternIndex == pattern.length)
            return true;//有效性检验：str到尾，pattern到尾，匹配成功
        if (strIndex != str.length && patternIndex == pattern.length)
            return false;//pattern先到尾，匹配失败
        //模式第2个是*，且字符串第1个跟模式第1个匹配,分3种匹配模式；如不匹配，模式后移2位
        if (patternIndex + 1 < pattern.length && pattern[patternIndex + 1] == '*') {
            if ((strIndex != str.length && pattern[patternIndex] == str[strIndex]) ||
                    pattern[patternIndex] == '.' && strIndex != str.length) {
                return matchCore(str, strIndex, pattern, patternIndex + 2) //模式后移2，视为x*匹配0个字符
                        || matchCore(str, strIndex + 1, pattern, patternIndex + 2)//视为模式匹配1个字符
                        || matchCore(str, strIndex + 1, pattern, patternIndex); //*匹配1个，再匹配str中的下一个
            } else {
                return matchCore(str, strIndex, pattern, patternIndex + 2);
            }
        }
        //模式第2个不是*，且字符串第1个跟模式第1个匹配，则都后移1位，否则直接返回false
        if ((strIndex != str.length && pattern[patternIndex] == str[strIndex]) || (pattern[patternIndex] == '.' && strIndex != str.length)) {
            return matchCore(str, strIndex + 1, pattern, patternIndex + 1);
        }
        return false;

    }

    //52. 表示数值的字符串
    private int indexTemp = 0;

    public boolean isNumeric(char[] str) {
        if (str == null)
            return false;
        boolean flag = scanInteger(str);
        if (indexTemp < str.length && str[indexTemp] == '.') {
            indexTemp++;
            flag = scanUnsignedInteger(str) || flag;
        }

        if (indexTemp < str.length && (str[indexTemp] == 'E' || str[indexTemp] == 'e')) {
            indexTemp++;
            flag = flag && scanInteger(str);
        }
        return flag && indexTemp == str.length;
    }

    public boolean scanInteger(char[] str) {
        if (indexTemp < str.length && (str[indexTemp] == '+' || str[indexTemp] == '-'))
            indexTemp++;
        return scanUnsignedInteger(str);
    }

    public boolean scanUnsignedInteger(char[] str) {
        int start = indexTemp;
        while (indexTemp < str.length && str[indexTemp] >= '0' && str[indexTemp] <= '9')
            indexTemp++;
        return start < indexTemp; //是否存在整数
    }

    //55. 字符流中第一个不重复的字符
    int[] hashtable = new int[256];
    StringBuffer sb = new StringBuffer();

    public void Insert(char ch) {
        sb.append(ch);
        if (hashtable[ch] == 0)
            hashtable[ch] = 1;
        else
            hashtable[ch] += 1;
    }

    public char FirstAppearingOnce() {
        char[] str = sb.toString().toCharArray();
        for (char c : str) {
            if (hashtable[c] == 1)
                return c;
        }
        return '#';
    }

    //56. 链表中环的入口结点
    public ListNode EntryNodeOfLoop(ListNode pHead) {
        ListNode meetingNode = meetingNode(pHead);
        if (meetingNode == null)
            return null;
        //得到环的节点个数
        int loopNumber = 1;
        ListNode node1 = meetingNode;  //令node为相遇点
        while (node1.next != meetingNode) {
            node1 = node1.next;
            loopNumber++;
        }

        node1 = pHead;
        for (int i = 0; i < loopNumber; i++) {
            node1 = node1.next;  //node走环的个数
        }
        ListNode node2 = pHead;
        while (node1 != node2) {
            node1 = node1.next;
            node2 = node2.next;
        }
        return node1;
    }

    private ListNode meetingNode(ListNode pHead) {
        if (pHead == null)
            return null;

        ListNode slow = pHead;
        ListNode fast = slow.next;

        while (fast != null && slow != null) {
            if (fast == slow)
                return fast;
            slow = slow.next; //慢指针一次走一步
            fast = fast.next;
            if (fast != null)//快指针一次走两步
                fast = fast.next;
        }
        return null;
    }

    // 57. 删除链表中重复的结点
    public ListNode deleteDuplication(ListNode pHead) {
        if (pHead == null || pHead.next == null)
            return pHead;

        ListNode pre = null;
        ListNode cur = pHead;

        while (cur != null) {
            ListNode next = cur.next;
            if (next != null && cur.val == next.val) {
                while (cur.val == next.val) {
                    cur = cur.next;
                }
                pre.next = cur.next;

            } else {
                cur = cur.next;
                if (pre == null)
                    pre = pHead;
                else
                    pre = pre.next;
            }
        }
        return pHead;
    }

    //58.给定一个二叉树和其中的一个结点，请找出中序遍历顺序的下一个结点
    public TreeLinkNode GetNext(TreeLinkNode pNode) {
        if (pNode == null) return null;
        if (pNode.right != null) {//如果有右子树，则找右子树的最左孩子
            pNode = pNode.right;
            while (pNode.left != null)
                pNode = pNode.left;
            return pNode;
        }

        while (pNode.next != null) {  //没有右子树时，则找到第一个当前节点是父节点左孩子的节点
            if (pNode.next.left == pNode)
                return pNode.next;//返回父节点
            pNode = pNode.next;
        }
        return null; //直到根节点仍没找到，则返回null
    }

    //59. 对称的二叉树
    boolean isSymmetrical(TreeNode pRoot) {
        return isSymmetric(pRoot, pRoot);//比较前序遍历序列和对称前序遍历序列是否相同
    }

    private boolean isSymmetric(TreeNode origin, TreeNode symme) {
        if (origin == null && symme == null) return true;
        if (origin == null || symme == null) return false;
        if (origin.val != symme.val) return false;
        return isSymmetric(origin.left, symme.right) && isSymmetric(origin.right, symme.left);
    }

    // 60. 把二叉树打印成多行（层序打印）
    public ArrayList<ArrayList<Integer>> Print(TreeNode pRoot) {
        ArrayList<ArrayList<Integer>> result = new ArrayList<ArrayList<Integer>>();
        if (pRoot == null)
            return result;
        Queue<TreeNode> nodes = new LinkedList<TreeNode>();
        ArrayList<Integer> layerList = new ArrayList<Integer>();//layerList完成层序遍历
        nodes.add(pRoot);//根入队
        int nextLevel = 0; //下一次节点数
        int toBePrinted = 1;//toBePrinted当前层中还没有打印节点节点数
        while (!nodes.isEmpty()) {
            TreeNode curNode = nodes.peek();
            layerList.add(curNode.val);
            if (curNode.left != null) {
                nodes.add(curNode.left);
                nextLevel++;//若一个节点有子节点，加队列，
            }
            if (curNode.right != null) {
                nodes.add(curNode.right);
                nextLevel++;
            }
            nodes.poll(); //删除一个节点
            --toBePrinted;
            if (toBePrinted == 0) {
                result.add(layerList);
                toBePrinted = nextLevel;
                nextLevel = 0;
                layerList = new ArrayList<Integer>();
            }
        }
        return result;
    }

    // 61. 按之字形顺序打印二叉树
    public ArrayList<ArrayList<Integer>> PrintWithZhi(TreeNode pRoot) {
        Stack<TreeNode> s1 = new Stack<TreeNode>();//s1存奇数层节点;    
        Stack<TreeNode> s2 = new Stack<TreeNode>();  //s2存偶数层节点
        int layer = 1; //标记行号(根为第一行)
        s1.push(pRoot); //根属于奇数行的结点(第一行)
        ArrayList<ArrayList<Integer>> list = new ArrayList<ArrayList<Integer>>();

        while (!s1.empty() || !s2.empty()) {
            if (layer % 2 == 1) {  //奇数,左孩子先入栈
                ArrayList<Integer> temp = new ArrayList<Integer>();
                while (!s1.empty()) {
                    TreeNode node = s1.pop();
                    if (node != null) {
                        temp.add(node.val);
                        System.out.print(node.val + " ");
                        if (node.left != null)
                            s2.push(node.left);
                        if (node.right != null)
                            s2.push(node.right);
                    }
                }
                if (!temp.isEmpty()) {
                    list.add(temp);
                    layer++;
                    System.out.println();
                }
            } else { //偶数行,右孩子先入栈
                ArrayList<Integer> temp = new ArrayList<Integer>();
                while (!s2.empty()) {
                    TreeNode node = s2.pop();
                    if (node != null) {
                        temp.add(node.val);
                        System.out.print(node.val + " ");
                        if (node.right != null)
                            s1.push(node.right);
                        if (node.left != null)
                            s1.push(node.left);
                    }
                }
                if (!temp.isEmpty()) {
                    list.add(temp);
                    layer++;
                    System.out.println();
                }
            }
        }
        return list;
    }

    //62. 序列化二叉树
    public int indexNum = -1;

    String Serialize(TreeNode root) {
        StringBuffer sb = new StringBuffer();
        if (root == null) {
            sb.append("#,");
            return sb.toString();
        }
        sb.append(root.val + ",");
        sb.append(Serialize(root.left));
        sb.append(Serialize(root.right));
        return sb.toString();
    }

    TreeNode Deserialize(String str) {
        indexNum++;
        int len = str.length();
        if (indexNum >= len) {
            return null;
        }
        String[] strr = str.split(",");
        TreeNode node = null;
        if (!strr[indexNum].equals("#")) {
            node = new TreeNode(Integer.valueOf(strr[indexNum]));
            node.left = Deserialize(str);
            node.right = Deserialize(str);
        }
        return node;
    }

    //63. 二叉搜索树的第k个结点（中序遍历）
    int index = 0; //计数器

    TreeNode KthNode(TreeNode root, int k) {
        if (root != null && k != 0) { //中序遍历寻找第k个
            TreeNode node = KthNode(root.left, k);
            if (node != null)
                return node;
            index++;
            if (index == k)
                return root;
            node = KthNode(root.right, k);
            if (node != null)
                return node;
        }
        return null;
    }

    //64. 数据流中的中位数（最小堆和最大堆实现）
    PriorityQueue<Integer> minHeap = new PriorityQueue<Integer>();
    PriorityQueue<Integer> maxHeap = new PriorityQueue<Integer>(11, new Comparator<Integer>() {
        @Override
        public int compare(Integer o1, Integer o2) {
            //PriorityQueue默认是小顶堆，实现大顶堆，需要反转默认排序器
            return o2.compareTo(o1);
        }
    });

    public void Insert(Integer num) {
        if (((maxHeap.size() + minHeap.size()) & 1) == 0) { //判偶数
            if (maxHeap.size() > 0 && num < maxHeap.peek()) {
                maxHeap.offer(num);
                num = maxHeap.poll();
            }
            minHeap.offer(num);
        } else {
            if (minHeap.size() > 0 && minHeap.peek() < num) {
                minHeap.offer(num);
                num = minHeap.poll();
            }
            maxHeap.offer(num);
        }
    }

    public Double GetMedian() {
        int size = maxHeap.size() + minHeap.size();
        if (size == 0)
            throw new RuntimeException("no available number!");
        Double median = 0.0;
        if ((size & 1) == 1)
            median = (double) minHeap.peek();
        else
            median = (minHeap.peek() + maxHeap.peek()) / 2.0;
        return median;
    }

    //65. 滑动窗口中元素最大值
    public ArrayList<Integer> maxInWindows(int[] num, int size) {
        ArrayList<Integer> ret = new ArrayList<>();
        if (num == null || num.length < size || size < 1) {
            return ret;
        }

        LinkedList<Integer> indexDeque = new LinkedList<>();
        for (int i = 0; i < size - 1; i++) {
            while (!indexDeque.isEmpty() && num[i] > num[indexDeque.getLast()]) {
                indexDeque.removeLast();
            }
            indexDeque.addLast(i);
        }

        for (int i = size - 1; i < num.length; i++) {
            while (!indexDeque.isEmpty() && num[i] > num[indexDeque.getLast()]) {
                indexDeque.removeLast();
            }
            indexDeque.addLast(i);
            if (i - indexDeque.getFirst() + 1 > size) {
                indexDeque.removeFirst();
            }
            ret.add(num[indexDeque.getFirst()]);
        }
        return ret;
    }

    //66. 矩阵中的路径
    public boolean hasPath(char[] matrix, int rows, int cols, char[] str) {
        boolean[] visited = new boolean[matrix.length];
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                if (searchFromHere(matrix, rows, cols, i, j, 0, str, visited))
                    return true;
            }
        }
        return false;
    }

    public boolean searchFromHere(char[] matrix, int rows, int cols, int r, int c, int index, char[] str, boolean[] visited) {
        if (r < 0 || r >= rows || c < 0 || c >= cols || matrix[r * cols + c] != str[index] || visited[r * cols + c])
            return false;
        if (index == str.length - 1) return true;
        visited[r * cols + c] = true;
        if (searchFromHere(matrix, rows, cols, r - 1, c, index + 1, str, visited) ||
                searchFromHere(matrix, rows, cols, r, c - 1, index + 1, str, visited) ||
                searchFromHere(matrix, rows, cols, r + 1, c, index + 1, str, visited) ||
                searchFromHere(matrix, rows, cols, r, c + 1, index + 1, str, visited))
            return true;
        visited[r * cols + c] = false;
        return false;
    }

    //67. 机器人的运动范围
    public int movingCount(int threshold, int rows, int cols) {
        int[][] flag = new int[rows][cols];
        return moving(threshold, rows, cols, flag, 0, 0);
    }

    public int moving(int threshold, int rows, int cols, int[][] flag, int i, int j) {
        if (threshold <= 0 || i >= rows || i < 0 || j >= cols || j < 0 || (flag[i][j] == 1) || (sum(i) + sum(j) > threshold)) {
            return 0;
        }
        flag[i][j] = 1;
        return moving(threshold, rows, cols, flag, i - 1, j)
                + moving(threshold, rows, cols, flag, i + 1, j)
                + moving(threshold, rows, cols, flag, i, j - 1)
                + moving(threshold, rows, cols, flag, i, j + 1)
                + 1;
    }

    public int sum(int i) {
        if (i == 0) {
            return i;
        }
        int sum = 0;
        while (i != 0) {
            sum += i % 10;
            i /= 10;
        }
        return sum;
    }

    //68.减绳子(法一：动态规划)
    public static int maxProductAfterCutting(int length) {
        if (length < 2)
            return 0;
        if (length == 2)
            return 1;
        if (length == 3)
            return 2;

        int[] products = new int[length + 1];
        products[0] = 0;
        products[1] = 1;
        products[2] = 2;
        products[3] = 3; //?
        int max = 0;
        for (int i = 4; i <= length; i++) {
            max = 0;
            for (int j = 1; j <= i / 2; ++j) {
                int proTemp = products[j] * products[i - j];
                if (max < proTemp)
                    max = proTemp;
                products[i] = max;
            }
        }
        max = products[length];
        return max;
    }

    //(法二：贪心算法)
    public static int maxProductAfterCutting2(int length) {
        if (length < 2)
            return 0;
        if (length == 2)
            return 1;
        if (length == 3)
            return 2;
        int timesOf3 = length / 3; //尽可能多减长度为3的绳子段
        if (length - timesOf3 * 3 == 1) //当长度为4时，2*2》1*3
            timesOf3 -= 1;

        int timesOf2 = (length - timesOf3 * 3) / 2;
        return (int) Math.pow(3, timesOf3) * (int) Math.pow(2, timesOf2);
    }

    //69. 数字序列中某一位的数字
    public static int digitAtIndex(int index) {
        if (index < 0)
            return -1;

        int digits = 1;
        while (true) {
            int number = countOfInteger(digits);
            if (index < number * digits)
                return digitAtIndex(index, digits);

            index -= digits * number;
            digits++;
        }
    }

    private static int digitAtIndex(int index, int digits) {
        int number = beginNumber(digits) + index / digits;
        int indexFromRight = digits - index % digits;
        for (int i = 1; i < indexFromRight; i++) {
            number /= 10;
        }
        return number % 10;
    }

    private static int beginNumber(int digits) {
        if (digits == 1)
            return 0;
        return (int) Math.pow(10, digits - 1);
    }

    private static int countOfInteger(int digits) { //m位的数字共有几个
        if (digits == 1)
            return 10;
        int count = (int) Math.pow(10, digits - 1);
        return 9 * count;
    }

    //70.把数字翻译成字符串0->a,1->b
    public static int getTranslationCount(int number) {
        if (number < 0)
            return 0;

        String str = Integer.toString(number);
        int length = str.length();
        int[] counts = new int[length];
        int count = 0; //从左到右翻译
        for (int i = length - 1; i >= 0; --i) {
            count = 0;
            if (i < length - 1)
                count = counts[i + 1];
            else
                count = 1;
            if (i < length - 1) {
                int digit1 = str.charAt(i) - '0';
                int digit2 = str.charAt(i + 1) - '0';
                int converted = digit1 * 10 + digit2;
                if (converted >= 10 && converted <= 25) {
                    if (i < length - 2)
                        count += counts[i + 2];
                    else
                        count += 1;
                }
            }
            counts[i] = count;
        }
        count = counts[0];
        return count;
    }

    //71. 礼物的最大价值（右移和下移）
    public static int getMaxValue(int[] values, int rows, int cols) {
        if (values == null || rows <= 0 || cols <= 0)
            return 0;
        int[] maxValues = new int[cols];
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                int left = 0;
                int up = 0;
                if (i > 0) up = maxValues[j];
                if (j > 0) left = maxValues[j - 1];

                maxValues[j] = Math.max(left, up) + values[i * cols + j];
            }
        }
        int maxValue = maxValues[cols - 1];
        return maxValue;
    }

    //72.股票的最大利润
    public static int maxDiff(int[] numbers, int length) {
        if (numbers == null || length < 2)
            return 0;
        int min = numbers[0]; //数组前n-1个数字的最小值
        int maxDiff = numbers[1] - min;//卖出价固定时，买入价越低，收益越大
        for (int i = 2; i < length; i++) {
            if (numbers[i - 1] < min)
                min = numbers[i - 1];

            int currentDiff = numbers[i] - min;
            if (currentDiff > maxDiff)
                maxDiff = currentDiff;
        }
        return maxDiff;
    }


}