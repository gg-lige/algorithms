
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

    // 最长不重复子串
    public int findUniqueLongestSubstring(String s) {
        int n = s.length();
        Set<Character> set = new HashSet<>();
        int ans = 0, i = 0, j = 0;
        while (i < n && j < n) {
            if (!set.contains(s.charAt(j))) {
                set.add(s.charAt(j++));
                ans = Math.max(ans, j - i);
            } else {
                set.remove(s.charAt(i++));
            }
        }
        return ans;
    }

    // 最长公共字串
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

    // 1.二维数组中的查找
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

    // 2.替换空格
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

    // 3.从尾到头打印链表
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

    // 4.重建二叉树(根据前序和中序序列)
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

    // 5.用两个栈实现队列
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

    // 6.旋转数组的最小数字
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

    // 7.斐波那契数列
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

    // 8.跳台阶，矩形覆盖
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

    // 12.调整数组顺序使奇数位于偶数前面
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

    // 13.链表中倒数第k个结点
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

    // 14.反转链表（循环）
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

    // 15.合并两个排序的链表
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

    // 16.树的子结构
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

    // 17.二叉树的镜像
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

    // 18.顺时针打印矩阵
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

    // 19.包含min函数的栈
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

    // 20.栈的压入、弹出序列
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

    // 21.从上往下打印二叉树
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

    // 22.某序列是否为二叉搜索树的后序遍历序列
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

    //23.二叉树中和为某一值的路径
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

    //24.复杂链表的复制
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

    //25.二叉搜索树转换为双向链表
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

    //26.打印字符串的所有排列
    public ArrayList<String> Permutation(String str) {
        List<String> res = new ArrayList<>();
        if (str != null && str.length() > 0) {
            PermutationHelper(str.toCharArray(), 0, res);
            Collections.sort(res);
        }
        return (ArrayList) res;
    }

    public void PermutationHelper(char[] cs, int i, List<String> list) {
        if (i == cs.length - 1) {
            String val = String.valueOf(cs);
            if (!list.contains(val))
                list.add(val);
        } else {
            for (int j = i; j < cs.length; j++) {
                swap(cs, i, j);
                PermutationHelper(cs, i + 1, list);
                swap(cs, i, j);
            }
        }
    }

    public void swap(char[] cs, int i, int j) {
        char temp = cs[i];
        cs[i] = cs[j];
        cs[j] = temp;
    }

    //27.数组中出现次数超过一半的数字
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

    //28. 最小的K个数
    public ArrayList<Integer> GetLeastNumbers_Solution(int[] input, int k) {
        ArrayList<Integer> result = new ArrayList<Integer>();
        int length = input.length;

        if (k > length || k == 0)
            return result;
        //利用优先队列建立最大堆
        PriorityQueue<Integer> maxHeap = new PriorityQueue<Integer>(k, new Comparator<Integer>() {
            @Override
            public int compare(Integer o1, Integer o2) {
                return o2.compareTo(o1);
            }
        });
        for (int i = 0; i < length; i++) {
            if (maxHeap.size() != k)//将input的前k个数存入maxHeap
            {
                maxHeap.offer(input[i]);
            } else if (maxHeap.peek() > input[i]) {
                maxHeap.poll();
                maxHeap.offer(input[i]);
            }
        }
        for (Integer i : maxHeap) {
            result.add(i);
        }

        return result;

    }

    //29.连续子数组的最大和
    public int FindGreatestSumOfSubArray(int[] array) {
        if (array.length == 0)//total记录累计值，maxSum记录和最大
            return 0;
        else {
            int total = array[0], maxSum = array[0];
            for (int i = 1; i < array.length; i++) {
                if (total >= 0)
                    total += array[i];
                else
                    total = array[i];
                if (total > maxSum)
                    maxSum = total;
            }
            return maxSum;
        }
    }

    //30.从1到n整数中1出现的次数
    public int NumberOf1Between1AndN_Solution(int n) {
        int count = 0;
        StringBuffer s = new StringBuffer();
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

    //31.把数组中的数按某顺序排成最小的数
    public String PrintMinNumber(int[] numbers) {
        int n = numbers.length;
        String s = "";
        ArrayList<Integer> list = new ArrayList<Integer>();

        for (int i = 0; i < n; i++) {
            list.add(numbers[i]);
        }
        Collections.sort(list, new Comparator<Integer>() {
            public int compare(Integer str1, Integer str2) {
                String s1 = str1 + "" + str2;
                String s2 = str2 + "" + str1;
                return s1.compareTo(s2);
            }
        });
        for (int j : list) {
            s += j;
        }
        return s;
    }

    //32.第n个丑数
    public int GetUglyNumber_Solution(int index) {
        if (index == 0) return 0;
        int n = 1, ugly = 1, min;
        Queue<Integer> q2 = new LinkedList<Integer>();
        Queue<Integer> q3 = new LinkedList<Integer>();
        Queue<Integer> q5 = new LinkedList<Integer>();
        q2.add(2);
        q3.add(3);
        q5.add(5);
        while (n != index) {
            ugly = Math.min(q2.peek(), Math.min(q3.peek(), q5.peek()));
            if (ugly == q2.peek()) {
                q2.add(ugly * 2);
                q3.add(ugly * 3);
                q5.add(ugly * 5);
                q2.poll();
            }
            if (ugly == q3.peek()) {
                q3.add(ugly * 3);
                q5.add(ugly * 5);
                q3.poll();
            }
            if (ugly == q5.peek()) {
                q5.add(ugly * 5);
                q5.poll();
            }
            n++;
        }
        return ugly;
    }

    //33.第一个只出现一次的字符的位置
    public int FirstNotRepeatingChar(String str) {
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

    //34. 数组中的逆序对个数 %1000000007
    public int InversePairs(int[] array) {
        if (array == null || array.length == 0) {
            return 0;
        }
        int[] copy = new int[array.length];
        for (int i = 0; i < array.length; i++) {
            copy[i] = array[i];
        }
        int count = InversePairsCore(array, copy, 0, array.length - 1);//数值过大求余
        return count;
    }

    private int InversePairsCore(int[] array, int[] copy, int low, int high) {
        if (low == high) {
            return 0;
        }
        int mid = (low + high) >> 1;
        int leftCount = InversePairsCore(array, copy, low, mid) % 1000000007;
        int rightCount = InversePairsCore(array, copy, mid + 1, high) % 1000000007;
        int count = 0;
        int i = mid; //前半段最后一个数字下标
        int j = high; //后半段最后一个数字下标
        int locCopy = high;
        while (i >= low && j > mid) {
            if (array[i] > array[j]) {
                count += j - mid;
                copy[locCopy--] = array[i--];
                if (count >= 1000000007)//数值过大求余
                {
                    count %= 1000000007;
                }
            } else {
                copy[locCopy--] = array[j--];
            }
        }
        for (; i >= low; i--) {
            copy[locCopy--] = array[i];
        }
        for (; j > mid; j--) {
            copy[locCopy--] = array[j];
        }
        for (int s = low; s <= high; s++) {
            array[s] = copy[s];
        }
        return (leftCount + rightCount + count) % 1000000007;
    }

    // 35. 两个链表中的第一个公共子节点
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

    //36. 二叉树的深度
    public int TreeDepth(TreeNode pRoot) {
        if (pRoot == null) {
            return 0;
        }
        int left = TreeDepth(pRoot.left);
        int right = TreeDepth(pRoot.right);
        return Math.max(left, right) + 1;
    }

    //37.数字在排序数组中出现的次数（二分查找法）
    public int GetNumberOfK(int[] array, int k) {
        int length = array.length;
        if (length == 0) {
            return 0;
        }
        int firstK = getFirstK(array, k, 0, length - 1);
        int lastK = getLastK(array, k, 0, length - 1, length);
        if (firstK != -1 && lastK != -1) {
            return lastK - firstK + 1;
        }
        return 0;
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

    private int getLastK(int[] array, int k, int start, int end, int length) {
        if (start > end) {
            return -1;
        }
        int mid = (start + end) >> 1;
        int middleData = array[mid];
        if (middleData == k) {
            if ((mid < length - 1 && array[mid + 1] != k) || mid == length - 1)
                return mid;
            else
                start = mid + 1;
        } else if (middleData < k)
            start = mid + 1;
        else
            end = mid - 1;
        return getLastK(array, k, start, end, length);
    }

    //38.二叉树是否是平衡二叉树
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

        if (Math.abs(left - right) > 1) {
            isBalanced = false;
        }
        return right > left ? right + 1 : left + 1;

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
        if (numberOfZero >= numberOfInterval) {
            return true;
        } else {
            return false;
        }
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

    //60. 对称的二叉树
    boolean isSymmetrical(TreeNode pRoot) {
        if (pRoot == null)
            return true;
        return isSymmetric(pRoot.left, pRoot.right);
    }

    private boolean isSymmetric(TreeNode left, TreeNode right) {
        if (left == null) return right == null;
        if (right == null) return false;
        if (left.val != right.val) return false; //左子树的右子树和右子树的左子树相同即可，采用递归
        return isSymmetric(left.left, right.right) && isSymmetric(left.right, right.left);
    }

    // 60. 把二叉树打印成多行（层序打印）
    ArrayList<ArrayList<Integer>> Print(TreeNode pRoot) {
        ArrayList<ArrayList<Integer>> result = new ArrayList<ArrayList<Integer>>();
        if (pRoot == null)
            return result;
        Queue<TreeNode> layer = new LinkedList<TreeNode>();
        ArrayList<Integer> layerList = new ArrayList<Integer>();
        layer.add(pRoot); //根入队
        int start = 0, end = 1; //第一层end为1；layerList完成层序遍历，用end记录每层结点数目
        while (!layer.isEmpty()) {
            TreeNode cur = layer.remove();  //出队
            layerList.add(cur.val);    //保存节点
            start++;    //记录当前存入的节点个数
            //节点左右孩子依次入队
            if (cur.left != null) {
                layer.add(cur.left);
            }
            if (cur.right != null) {
                layer.add(cur.right);
            }
            if (start == end) {
                end = layer.size();   //更新end:该层节点的个数
                start = 0;    //start重新计数
                result.add(layerList);
                layerList = new ArrayList<Integer>();  //重新初始化新的layerList，存放新的一层的节点值
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
        if (root != null) { //中序遍历寻找第k个
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
    int count;
    PriorityQueue<Integer> minHeap = new PriorityQueue<Integer>();
    PriorityQueue<Integer> maxHeap = new PriorityQueue<Integer>(11, new Comparator<Integer>() {
        @Override
        public int compare(Integer o1, Integer o2) {
            //PriorityQueue默认是小顶堆，实现大顶堆，需要反转默认排序器
            return o2.compareTo(o1);
        }
    });

    public void Insert(Integer num) {
        count++;
        if ((count & 1) == 0) { // 判断偶数的高效写法
            if (!maxHeap.isEmpty() && num < maxHeap.peek()) {
                maxHeap.offer(num);
                num = maxHeap.poll();
            }
            minHeap.offer(num);
        } else {
            if (!minHeap.isEmpty() && num > minHeap.peek()) {
                minHeap.offer(num);
                num = minHeap.poll();
            }
            maxHeap.offer(num);
        }
    }

    public Double GetMedian() {
        if (count == 0)
            throw new RuntimeException("no available number!");
        double result;
        //总数为奇数时，大顶堆堆顶就是中位数
        if ((count & 1) == 1)
            result = maxHeap.peek();
        else
            result = (minHeap.peek() + maxHeap.peek()) / 2.0;
        return result;
    }

    //65. 滑动窗口中元素最大值
    public ArrayList<Integer> maxInWindows(int[] num, int size) {
        ArrayList<Integer> ret = new ArrayList<>();
        if (num == null) {
            return ret;
        }
        if (num.length < size || size < 1) {
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

}