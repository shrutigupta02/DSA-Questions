1. Reverse string:
```
class Solution {
    public void reverseString(char[] s) {
        int n = s.length;
        for(int i = 0; i <  n/2; i++){
            char temp = s[i];
            s[i] = s[n - i - 1];
            s[n - i - 1] = temp;
        }
    }
}
```

2. Valid Anagram:
```
class Solution {
    public boolean isAnagram(String s, String t) {
        if (s.length()!=t.length()){
            return false;
        }
        Map<Character, Integer> mp = new HashMap<>();
        for(char c: s.toCharArray()){
            mp.put(c, mp.getOrDefault(c, 0) + 1);
        }
        for(char c : t.toCharArray()){
            if(!mp.containsKey(c) || mp.get(c) <= 0) return false;
            mp.put(c, mp.get(c) - 1);
        }
        return true;
    }
}
```

3. Reverse words III:
```
class Solution {
    public String reverseWords(String s) {
        String[] words = s.split(" ");
        StringBuilder result = new StringBuilder();

        for (int i = 0; i < words.length; i++) {
            result.append(new StringBuilder(words[i]).reverse());
            if (i != words.length - 1) result.append(" ");
        }

        return result.toString();
    }
}
```

4. Haystack and needle, find first occurrence:
```
class Solution {
    public int strStr(String haystack, String needle) {
        int hLen = haystack.length();
        int nLen = needle.length();
        if(haystack.equals(needle)) return 0;
        for(int i = 0; i <= hLen - nLen; i++){
            int j = 0;
            while(j < nLen && needle.charAt(j) == haystack.charAt(i+j))
                j++;
            if(j == nLen) return i;  
        }
        return -1;
    }
}
```

5. Longest common prefix:
```
class Solution {
    public String longestCommonPrefix(String[] strs) {
        String prefix = strs[0];
        for(String word: strs){
            int i = 0;
            while(i < prefix.length() && i < word.length() && word.charAt(i) == prefix.charAt(i))
                i++;
            prefix = prefix.substring(0, i);
        }
        return prefix;
    }
}
```

6. Binary Search:
```
class Solution {
    public int search(int[] nums, int target) {
        int low = 0;
        int high = nums.length - 1;

        while (low <= high) {
            int mid = low + (high - low) / 2;

            if (nums[mid] == target) {
                return mid;
            }
            if (nums[mid] > target) {
                high = mid - 1;
            } else {
                low = mid + 1;
            }
        }
        return -1;
    }
}
```

7. Find index:
```
class Solution {
    public int searchInsert(int[] nums, int target) {
        int index = nums.length;
        int low = 0;
        int high = nums.length - 1;
        while(low <= high){
            int mid = (low+high) / 2;
            if(target <= nums[mid]){
                index = mid;
                high = mid - 1;
            }
            else low = mid+1;
        }
        return index;
    }
}
```

8. Find min using binary search:
```
class Solution {
    public int findMin(int[] nums) {
        if(nums.length == 1) return nums[0];
        int low = 0, high = nums.length - 1;
        int res = nums[0];
        while(low<=high){
            int mid = (low+high) / 2;
            if(nums[mid]>=nums[low]){
                res = Math.min(res, nums[low]);
                low = mid + 1;
            }
            else{
                res = Math.min(res, nums[mid]);
                high = mid - 1;
            }
        }
        return res;
    }
}
```

9. Find number of times array is rotated:
```
public static int findKRotation(int[] arr) {
        int low = 0, high = arr.length - 1;
        int ans = Integer.MAX_VALUE;
        int index = -1;
        while (low <= high) {
            int mid = (low + high) / 2;
            //search space is already sorted
            //then arr[low] will always be
            //the minimum in that search space:
            if (arr[low] <= arr[high]) {
                if (arr[low] < ans) {
                    index = low;
                    ans = arr[low];
                }
                break;
            }
            //if left part is sorted:
            if (arr[low] <= arr[mid]) {
                // keep the minimum:
                if (arr[low] < ans) {
                    index = low;
                    ans = arr[low];
                }
                // Eliminate left half:
                low = mid + 1;
            } else { //if right part is sorted:
                // keep the minimum:
                if (arr[mid] < ans) {
                    index = mid;
                    ans = arr[mid];
                }
                // Eliminate right half:
                high = mid - 1;
            }
        }
        return index;
    }
```

10. Convert array to LL:
```
private static Node convertArr2DLL(int[] arr) {
        // Create the head node with the first element of the array
        Node head = new Node(arr[0]);
        // Initialize 'prev' to the head node
        Node prev = head;

        for (int i = 1; i < arr.length; i++) {
            // Create a new node with data from the array and set its 'back' pointer to the previous node
            Node temp = new Node(arr[i], null, prev);
            // Update the 'next' pointer of the previous node to point to the new node
            prev.next = temp;
            // Move 'prev' to the newly created node for the next iteration
            prev = temp;
        }
        // Return the head of the doubly linked list
        return head;
    }
```

11. Reverse DLL:
```
public Node reverseDLL(Node head) {
    Node curr = head;
    Node temp = null;

    while (curr != null) {
        // Swap next and prev
        temp = curr.prev;
        curr.prev = curr.next;
        curr.next = temp;
        
        // Move to next node in original direction (which is curr.prev now)
        curr = curr.prev;
    }

    // At the end, temp will be the new head's prev
    if (temp != null) {
        head = temp.prev;  // Since temp is one step past the new head
    }

    return head;
}
```

12. Find middle of LL:
```
class Solution {
    public ListNode middleNode(ListNode head) {
        ListNode tortoise = head;
        ListNode hare = head;
        while(hare!=null && hare.next!=null){
            hare = hare.next.next;
            tortoise = tortoise.next;
        }
        return tortoise;
    }
}
```

13. Reverse Single LL:
```
class Solution {
    public ListNode reverseList(ListNode head) {
        ListNode temp = head;
        ListNode back = null, front = null;
        while(temp!=null){
            front = temp.next;
            temp.next = back;
            back = temp;
            temp = front;
        }
        return back;
    }
}
```

14. Loop in LL: approach 1 is use a map<node, int>, approach 2: use tortoise and hare if both equal then yes cycle exists.
15. LL palindrome: reverse LL + middle element 
16. Sort LL merge sort code
17. pow(x,y):
```
class Solution {
    public double myPow(double x, int n) {
        if(n==0) return 1;
        double ans = 1.0;
        long nn = n;
        if(n < 0){
            nn*=-1;
        }
        while(nn > 0){
            if(nn%2 == 1){
                ans *= x;
                nn--;
            }
            else{
                x*=x;
                nn /= 2;
            }
        }
        if(n < 0) ans = 1 / ans;
        return ans;
    }
}
```

18. reverse stack using recursion:
```
class Solution {
    public void reverseStack(Stack<Integer> st) {
        if(st.empty()) return;
        int top = st.pop();
        reverseStack(st);
        insertAtBottom(st, top);
    }

    public void insertAtBottom(Stack<Integer> st, int element){
        if(st.empty()){
            st.push(element);
            return;
        }
        int temp = st.pop();
        insertAtBottom(st, element);
        st.push(temp);
    }
}
```

19. Remove duplicates from sorted array:
```
static int removeDuplicates(int[] arr) {
        int i = 0;
        for (int j = 1; j < arr.length; j++) {
            if (arr[i] != arr[j]) {
                i++;
                arr[i] = arr[j];
            }
        }
        return i + 1;
    }
```

20. Rotate array by k places:
    - rotate left: reverse first k, reverse last n-k, reverse entire array
    - rotate right: reverse first n-k, reverse last k, reverse all

21. Move zeroes to end: initialise j pointer and using a loop, get j pointer to point at first zero in array, start another loop then starting from j+1 and keep swapping i and j until a[i]!=0
```java
public static int[] moveZeros(int n, int[] a) {
        int j = -1;
        //place the pointer j:
        for (int i = 0; i < n; i++) {
            if (a[i] == 0) {
                j = i;
                break;
            }
        }

        //no non-zero elements:
        if (j == -1) return a;

        //Move the pointers i and j
        //and swap accordingly:
        for (int i = j + 1; i < n; i++) {
            if (a[i] != 0) {
                //swap a[i] & a[j]:
                int tmp = a[i];
                a[i] = a[j];
                a[j] = tmp;
                j++;
            }
        }
        return a;
    }
```

22. Implement stack:
```java
class stack {
    int size = 10000;
    int arr[] = new int[size];
    int top = -1;
    void push(int x) {
        top++;
        arr[top] = x;
    }
    int pop() {
        int x = arr[top];
        top--;
        return x;
    }
    int top() {
        return arr[top];
    }
    int size() {
        return top + 1;
    }
}
```

23. Implement stack using two queues:
```java
class MyStack {
    Queue<Integer> q1;
    Queue<Integer> q2;
    public MyStack() {
        q1 = new ArrayDeque<>();
        q2 = new ArrayDeque<>();
    }
    public void push(int x) {
        q2.add(x);
        while(!empty()){
            q2.add(q1.remove());
        }
        Queue<Integer> temp = q1;
        q1 = q2;
        q2 = temp;
    }
    public int pop() {
        return q1.poll();
    }
    public int top() {
        return q1.peek();
    }
    public boolean empty() {
        return q1.isEmpty();
    }
}
```

24. Implement queue using 2 stacks:
```java
class MyQueue {
    Stack<Integer> st1 = new Stack<>();
    Stack<Integer> st2 =new Stack<>();

    public MyQueue() {

    }

    public void push(int x) {
        st1.push(x);
    }

    public int pop() {
        if(st2.isEmpty()) {
            while(!st1.isEmpty())  {
                st2.push(st1.pop());
            }
        }
        return st2.pop();
    }

    public int peek() {
         if(st2.isEmpty()) {
            while(!st1.isEmpty())  {
                st2.push(st1.pop());
            }
        }
        return st2.peek();
    }

    public boolean empty() {
        return st1.isEmpty() && st2.isEmpty();
    }
}
```

25. Check parentheses:
```java
class Solution {
    public boolean isValid(String s) {
        Stack<Character> brackets = new Stack<>();
        for(char c: s.toCharArray()){
            switch(c){
                case '(':
                brackets.push(')');
                break;
                case '{':
                brackets.push('}');
                break;
                case '[':
                brackets.push(']');
                break;
                default:
                    if(brackets.isEmpty() || brackets.peek() != c) 
                    return false;
                    else brackets.pop();
            }
        }
        return brackets.isEmpty();
    }
}
```

26. Implement minstack:
```java
class MinStack {
    Stack<Integer> minStack;
    Stack<Integer> min;
    public MinStack() {
        minStack = new Stack<>();
        min = new Stack<>();
    }
    public void push(int val) {
        if(min.isEmpty() || min.peek() >= val){
            min.push(val);
        }
        minStack.push(val);
    }
    public void pop() {
        if(minStack.isEmpty()) return;
        int temp = minStack.pop();
        if(temp == min.peek()) min.pop();
    }
    public int top() {
        if(minStack.isEmpty()) return -1;
        return minStack.peek();
    }
    public int getMin() {
        if(min.isEmpty()) return -1;
        return min.peek();
    }
}
```

27. Infix to postfix:
```java
static int Prec(char ch) {
    switch (ch) {
    case '+':
    case '-':
      return 1;

    case '*':
    case '/':
      return 2;

    case '^':
      return 3;
    }
    return -1;
  }
  static String infixToPostfix(String exp) {
    String result = new String("");
    Stack < Character > stack = new Stack < > ();
    for (int i = 0; i < exp.length(); ++i) {
      char c = exp.charAt(i);
      // If the scanned character is an
      // operand, add it to output.
      if (Character.isLetterOrDigit(c))
        result += c;
      // If the scanned character is an '(',
      // push it to the stack.
      else if (c == '(')
        stack.push(c);
      // If the scanned character is an ')',
      // pop and output from the stack
      // until an '(' is encountered.
      else if (c == ')') {
        while (!stack.isEmpty() && stack.peek() != '(') 
            result += stack.pop();
        stack.pop();
      } 
      else // an operator is encountered
      {
        while (!stack.isEmpty() && Prec(c) <=
          Prec(stack.peek())) {

          result += stack.pop();
        }
        stack.push(c);
      }
    }
    // pop all the operators from the stack
    while (!stack.isEmpty()) {
      if (stack.peek() == '(')
        return "Invalid Expression";
      result += stack.pop();
    }
    return result;
  }
```

28. Infix to prefix: jaha bhi prefix padho vaha par reverse karo 
```java
class Solution{
	public static String infixToPrefix(String infix) {
    // Step 1: Reverse the infix expression
    String reversedInfix = reverse(infix);
    
    // Step 2: Replace '(' with ')' and vice versa
    String modifiedInfix = reverseParentheses(reversedInfix);
    
    // Step 3: Convert modified infix to postfix
    String postfix = infixToPostfix(modifiedInfix);
    
    // Step 4: Reverse the postfix to get prefix
    String prefix = reverse(postfix);
    
    return prefix;
	}
	private static String reverseParentheses(String str) {
    StringBuilder result = new StringBuilder();
    for (char c : str.toCharArray()) {
        if (c == '(') {
            result.append(')');
        } else if (c == ')') {
            result.append('(');
        } else {
        result.append(c);
        }
    }
    return result.toString();
	}
}
```

29. Prefix to infix: reverse kara hai for loop me dekh
```java
public static String prefixToInfix(String prefix) {
        Stack<String> stack = new Stack<>();
        
        // Read from right to left
        for (int i = prefix.length() - 1; i >= 0; i--) {
            char c = prefix.charAt(i);
            
            if (isOperand(c)) {
                stack.push(String.valueOf(c));
            } else if (isOperator(c)) {
                // Pop two operands
                if (stack.size() < 2) {
                    throw new IllegalArgumentException("Invalid prefix expression");
                }
                String operand1 = stack.pop();
                String operand2 = stack.pop();
                
                // Create infix expression: (operand1 operator operand2)
                String infixExpr = "(" + operand1 + c + operand2 + ")";
                stack.push(infixExpr);
            }
        }
        
        if (stack.size() != 1) {
            throw new IllegalArgumentException("Invalid prefix expression");
        }
        
        return stack.pop();
}
```

30. Postfix to infix: same as above without reverse
```java
public static String postfixToInfix(String postfix) {
        Stack<String> stack = new Stack<>();
        
        // Read from left to right
        for (char c : postfix.toCharArray()) {
            if (isOperand(c)) {
                stack.push(String.valueOf(c));
            } else if (isOperator(c)) {
                // Pop two operands
                if (stack.size() < 2) {
                    throw new IllegalArgumentException("Invalid postfix expression");
                }
                String operand2 = stack.pop();
                String operand1 = stack.pop();
                
                // Create infix expression: (operand1 operator operand2)
                String infixExpr = "(" + operand1 + c + operand2 + ")";
                stack.push(infixExpr);
            }
        }
        
        if (stack.size() != 1) {
            throw new IllegalArgumentException("Invalid postfix expression");
        }
        
        return stack.pop();
}
```

31. Prefix to postfix: phirse reverse
```java
public static String prefixToPostfix(String prefix) {
        Stack<String> stack = new Stack<>();
        
        // Read from right to left
        for (int i = prefix.length() - 1; i >= 0; i--) {
            char c = prefix.charAt(i);
            
            if (isOperand(c)) {
                stack.push(String.valueOf(c));
            } else if (isOperator(c)) {
                // Pop two operands
                if (stack.size() < 2) {
                    throw new IllegalArgumentException("Invalid prefix expression");
                }
                String operand1 = stack.pop();
                String operand2 = stack.pop();
                
                // Create postfix expression: operand1 operand2 operator
                String postfixExpr = operand1 + operand2 + c;
                stack.push(postfixExpr);
            }
        }
        
        if (stack.size() != 1) {
            throw new IllegalArgumentException("Invalid prefix expression");
        }
        
        return stack.pop();
}
```

32. Postfix to prefix:
```java
class Solution{
    public static String postfixToPrefix(String postfix) {
        Stack<String> stack = new Stack<>();
        // Read from left to right
        for (char c : postfix.toCharArray()) {
            if (isOperand(c)) {
                stack.push(String.valueOf(c));
            } else if (isOperator(c)) {
                // Pop two operands
                if (stack.size() < 2) {
                    throw new IllegalArgumentException("Invalid postfix expression");
                }
                String operand2 = stack.pop();
                String operand1 = stack.pop();
                // Create prefix expression: operator operand1 operand2
                String prefixExpr = c + operand1 + operand2;
                stack.push(prefixExpr);
            }
        }
        if (stack.size() != 1) {
            throw new IllegalArgumentException("Invalid postfix expression");
        }
        return stack.pop();
    }
}
```

33. Next greater element:
```java
class Solution {
    public int[] nextGreaterElement(int[] nums1, int[] nums2) {
        Stack<Integer> stack = new Stack<>();
        Map<Integer, Integer> map = new HashMap<>();
        for(int num : nums2){
            while(!stack.isEmpty() && stack.peek() < num){
                map.put(stack.pop(), num);
            }
            stack.push(num);
        }
        int[] res = new int[nums1.length];
        for(int i = 0; i < nums1.length; i++){
            res[i] = map.getOrDefault(nums1[i], -1);
        }
        return res;
    }
}
```

34. Next greater element 2: **Circular array me use index as i%n and i < n\*2-1***
```java
class Solution {
    public int[] nextGreaterElements(int[] nums) {
        Stack<Integer> stack = new Stack<>();
        int[] res = new int[nums.length];
        Arrays.fill(res, -1);
        int n = nums.length;
        for(int i = 0 ; i < n*2-1; i++){
            int num = nums[i % n];
            while(!stack.isEmpty() && nums[stack.peek()] < num){
                int index = stack.pop();
                res[index] = num;
            }
            if(i < n){
                stack.push(i);
            }
        }
        return res;
    }
 }
```

35. Reverse Polish Notation:
```java
class Solution {
    public int evalRPN(String[] tokens) {
        Stack<Integer> stack = new Stack<>();
        for(String s : tokens){
            if(isOperator(s)){
                int right = stack.pop();
                int left = stack.pop();
                int ans = operate(left, right, s);
                stack.push(ans);
            }
            else{
                stack.push(Integer.parseInt(s));
            }
        }
        return stack.pop();
    }
    private boolean isOperator(String c){
        return c.equals("+") || c.equals("-") || c.equals("*") || c.equals("/");
    }
    private int operate(int a, int b, String c){
            if(c.equals("+"))
                return a+b;
            if(c.equals("-"))
                return a-b;
            if(c.equals("*"))
                return a*b;
            if(c.equals("/")){
                if(b == 0)
                    return 0;
                return a/b;
            }
        return -1;
    }
 }
```

36. Sieve of eratosthenes:
```java
static int[] sieve(int n) {
        
        // creation of boolean array
        boolean[] prime = new boolean[n + 1];
        for (int i = 0; i <= n; i++) {
            prime[i] = true;
        }
 
        for (int p = 2; p * p <= n; p++) {
            if (prime[p]) {
                // marking as false
                for (int i = p * p; i <= n; i += p)
                    prime[i] = false;
            }
        }
 
        // Count number of primes
        int count = 0;
        for (int p = 2; p <= n; p++) {
            if (prime[p])
                count++;
        }
 
        // Store primes in an array
        int[] res = new int[count];
        int index = 0;
        for (int p = 2; p <= n; p++) {
            if (prime[p])
                res[index++] = p;
        }
 
        return res;
    }
```

37. Happy Number:
```java
class Solution {
    public boolean isHappy(int n) {
        HashSet<Integer> seen = new HashSet<>();
        int sum = n;
        while(sum!=1 && !seen.contains(sum)){
            seen.add(sum);
            sum = getSum(sum);
        }
        return sum==1;
    }
    private int getSum(int n){
        int sum = 0;
        while(n>0){
            int digit1 = n%10;
            digit1 *= digit1;
            sum+=digit1;
            n/=10;
        }
        return sum;
    }
 }
```

38. BFS: uses queue
```java
public ArrayList<Integer> bfsOfGraph(int V, 
    ArrayList<ArrayList<Integer>> adj) {
        
        ArrayList < Integer > bfs = new ArrayList < > ();
        boolean vis[] = new boolean[V];
        Queue < Integer > q = new LinkedList < > ();

        q.add(0);
        vis[0] = true;
        
        while(!q.isEmpty()){
	        int currentNode = q.poll();
	        bfs.add(currentNode);
	        
	        //node represents all the branches of the current node
	        //and we only add the branches that we have not visited 
	        //via another node hence we check if vis[node] = false
	        for(int node : adj.get(currentNode)){
		        if(vis[node] == false){
			        vis[node] = true;
			        q.add(node);
		        }
	        }
        }

        return bfs;
    }
```

39. DFS: use recursion or stack
```java
public static void backtrack(int node, List<List<Integer>> adj, boolean[] visited, List<Integer> current){

	current.add(node);
	visited[node] = true;
	for(int curr : adj.get(node)){
		if(visited[curr] == false){
			backtrack(curr, adj, visited, current);
		}
	}
 }

 public ArrayList<Integer> dfs(int V, List<List<Integer>> adj){
	 ArrayList<Integer> current = new ArrayList<>();
	 boolean[] visited = new boolean[V+1];
	 visited[0] = true;
	 backtrack(0, adj, visited, current);
	 return current;
 }
```

40. Implement heap using arraylist:
```java
import java.util.*;
class Heap{
    private ArrayList<Integer> arr;
    public Heap(){
        arr = new ArrayList<>();
    }
    private void swap(int a, int b){
        // a and b are indices
        int temp = arr.get(a);
        arr.set(a, arr.get(b));
        arr.set(b, temp);
    }
    
    private int parent(int index){
        index = index - 1; //because arraylist is 0 indexed
        return index/2;
    }
    
    private int left(int index){
        //returns left child
        return index * 2 + 1;
    }
    
    private int right(int index){
        return index*2+2;
    }
    
    public void insert(int num){
        arr.add(num);
        upheap(arr.size()-1);
        System.out.println(arr);
    }
    
    private void upheap(int index){
        if(index == 0) return;
        if(arr.get(index) < arr.get(parent(index))){
            swap(index, parent(index));
            upheap(parent(index));
        }
    }
    
    public void remove(){
        if(arr.isEmpty()) return;
        int elem = arr.get(0);
        if(!arr.isEmpty()){
            int last = arr.remove(arr.size()-1);
            arr.set(0, last);
            downheap(0);
        }
        System.out.println(arr);
    }
    
    private void downheap(int index){
        int left = left(index);
        int right = right(index);
        int temp = arr.get(index);
        int min = index;
        if(left < arr.size() && temp > arr.get(left)) min = left;
        if(right < arr.size() && temp > arr.get(right)) min = right;
        if(index != min){
            swap(index, min);
            downheap(min);
        }
    }
 }
```

41. Heap Sort:
```java
// implement in heap class defined above because everytime you remove you 
// get the smallest element in the heap
public ArrayList<Integer> heapSort(){
	ArrayList<Integer> res = new ArrayList<>();
	while(!arr.isEmpty()){
		res.add(this.remove());
	}
	return res;
 }
```

42. Check if max heap:
```java
public static boolean isMaxHeap(int[] arr) {
        int n = arr.length;
        for (int i = 0; i < n / 2; i++) {
            int left = 2 * i + 1;
            int right = 2 * i + 2;
            
            if (left < n && arr[i] < arr[left]) return false;
            if (right < n && arr[i] < arr[right]) return false;
        }
        return true;
    }
```

43. build max or min heap from unsorted array: call downheap function from non leaf nodes i.e. n/2-1;
44. Path Sum:
```java
class Solution {
    public boolean hasPathSum(TreeNode root, int targetSum) {
        if (root == null) {
            return false;
        }
        
        if (root.left == null && root.right == null) {
            return targetSum == root.val;
        }
        
        boolean leftSum = hasPathSum(root.left, targetSum - root.val);
        boolean rightSum = hasPathSum(root.right, targetSum - root.val);
        
        return leftSum || rightSum;
    }
 }           
```
45. Same tree
```java
class Solution {
    public boolean isSameTree(TreeNode p, TreeNode q) {
        if(p == null && q == null) return true;
        if(p == null || q == null) return false;
        if(p.val != q.val) return false;

        boolean leftSame = isSameTree(p.left, q.left);
        boolean rightSame = isSameTree(p.right, q.right);

        return leftSame && rightSame;
    }
 }
```
46. 