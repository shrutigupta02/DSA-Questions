1. Count digits: //10
2. Reverse num: %10 and revnum = revnum \* 10 + digit
3. Palindrome: reverse and equate
4. GCD or HCF: 
	   approach: when you subtract num1 from num2, their gcd remains the same, hence continue subtracting the lower number from greater number until one becomes zero, the other number left will be the gcd
5. Armstrong num: num equal to the sum of its digits raised to the power of number of digits
	   intuitive approach hee hai karle bhay
6. Print All divisors: 
	   approach: to make it optimal, notice that the divisors always repeat (in reverse) after the square root of the number
7. Check for prime: use the same approach as above
8. Recursion: two ways: parametrised and functional
9. Reverse array: space optimised: two pointer -> swap L=0, L++ and R=n-1, R--
	   recursive way: use the same technique discussed above but using recursion
10. Hashing: use dictionary {}
11. Freq of most freq  element:
```
class Solution {
    public int maxFrequency(int[] nums, int k) {
        // Sort array to group similar elements together
        Arrays.sort(nums);
        
        int left = 0;
        long windowSum = 0;  // Use long to prevent overflow
        int maxFrequency = 1;
        
        // Sliding window approach
        for (int right = 0; right < nums.length; right++) {
            // Add current element to window
            windowSum += nums[right];
            
            // Calculate cost to make all elements in window equal to nums[right]
            // Cost = target * windowSize - currentSum
            long target = nums[right];
            long windowSize = right - left + 1;
            long costToMakeAllEqual = target * windowSize - windowSum;
            
            // If cost exceeds k, shrink window from left
            while (costToMakeAllEqual > k) {
                windowSum -= nums[left];
                left++;
                windowSize = right - left + 1;
                costToMakeAllEqual = target * windowSize - windowSum;
            }
            
            // Update maximum frequency
            maxFrequency = Math.max(maxFrequency, (int)windowSize);
        }
        
        return maxFrequency;
    }
}
```
1. Sorts: Merge sort code
```
def merge_sort(arr: list, l, r) -> list:
    if l >= r:
        return
    med = (l+r)//2
    merge_sort(arr, l, med)
    merge_sort(arr, med+1, r)
    merge(arr, l, med, r)
    return arr    
def merge(arr: list, low, med, high) -> list:
    left = low
    right = med + 1
    temp = [0] * (high - low + 1)

    i = 0
    
    while left <= med and right <= high:
        if(arr[left] < arr[right]):
            temp[i] = arr[left]
            left += 1
        else:
            temp[i] = arr[right]
            right+=1
        i+=1
        
    while left <= med:
        temp[i] = arr[left]
        left+=1
        i+=1
        
    while right <= high:
        temp[i] = arr[right]
        right += 1
        i += 1
    
    for i in range(len(temp)):
        arr[low + i] = temp[i]
print(merge_sort([3,2,6,1,8], 0, 4))
```

12. Quick Sort code:
```
def partition(arr, low, high):
    pivot = arr[low]
    i = low
    j = high
    while(i < j):
        while(i <= high - 1 and arr[i] <= pivot):
            i+=1
        while(j >= low + 1 and arr[j] > pivot):
            j-=1
        if(i < j):
            arr[i], arr[j] = arr[j], arr[i]
        
    arr[j], arr[low] = arr[low], arr[j]
    
    return j
def quick_sort(arr, low, high):
    if low < high:
        pivot_index = partition(arr, low, high)
        quick_sort(arr, low, pivot_index-1)
        quick_sort(arr, pivot_index + 1, high)
arr = [3, 1, 5, 7, 2, 4, 8]
quick_sort(arr, 0, 6)
print(arr)
```

13. Rotate array:
	rotate to left: reverse k elements, then reverse n-k elements, reverse entire array
	rotate to right: reverse n-k elements, reverse k elements, reverse entire array
14. Move zeroes:
```
def move_zeroes(arr: list) -> list:
    n = len(arr)
    if n <= 1: return arr
    j = 0
    while j<n-1 and arr[j] != 0:
        j+=1
    for i in range(j, n):
        if arr[i] != 0:
            arr[i], arr[j] = arr[j], arr[i]
            j+=1
    return arr
print(move_zeroes([2, 0, 1, 0, 3, 8, 0]))
```
15. Union of two arrays: optimised approach use two pointers
16. Find missing number in array: calculate sum of n terms using AP formula: n*(n + 1) // 2 and then subtract sum of input array from it
17. **subarray of sum K:**
```
def getLongestSubarray(a: [int], k: int) -> int:
    n = len(a) # size of the array.
    preSumMap = {}
    Sum = 0
    maxLen = 0
    for i in range(n):
        # calculate the prefix sum till index i:
        Sum += a[i]

        # if the sum = k, update the maxLen:
        if Sum == k:
            maxLen = max(maxLen, i + 1)

        # calculate the sum of remaining part i.e. x-k:
        rem = Sum - k

        # Calculate the length and update maxLen:
        if rem in preSumMap:
            length = i - preSumMap[rem]
            maxLen = max(maxLen, length)

        # Finally, update the map checking the conditions:
        if Sum not in preSumMap:
            preSumMap[Sum] = i

    return maxLen
```

18. 2Sum: add elements in hashmap with their index and check if their difference from target exists in map
19. Dutch National Flag, sort colors, sort 0,1,2: 
approach 1 - > use three counts for 0, 1, and 2. then run loops from 0 - count0, count0 - count0+count1, count0+count1 - count0+count1+count2
app 2 -> maintain three pointers: low, mid, high and :
```
def sortColors(self, arr: List[int]) -> None:
	low, mid = 0, 0
	high = len(arr) - 1
	while mid <= high:
	if arr[mid] == 0: #swap mid and low
	arr[low], arr[mid] = arr[mid], arr[low] 
	low += 1
	mid += 1
	
	elif arr[mid] == 1:
	mid += 1
	
	else: #swap mid and high
	arr[mid], arr[high] = arr[high], arr[mid]
	high -= 1
```

20. Majority element: **Moore Voting Algo**
approach:
```
def majorityElement(self, nums: List[int]) -> int:
	count = 0
	element = nums[0]
	for num in nums:
	if count == 0:
	element = num
	count += 1
	elif element == num:
	count += 1
	else:
	count -= 1
	return element
```

21. Kadane's algo // Max subarray:
```java
public static long maxSubarraySum(int[] arr, int n) {
        long maxi = Long.MIN_VALUE; // maximum sum
        long sum = 0;

        int start = 0;
        int ansStart = -1, ansEnd = -1;
        for (int i = 0; i < n; i++) {

            if (sum == 0) start = i; // starting index

            sum += arr[i];

            if (sum > maxi) {
                maxi = sum;

                ansStart = start;
                ansEnd = i;
            }

            // If sum < 0: discard the sum calculated
            if (sum < 0) {
                sum = 0;
            }
        }
```

21. buy and sell stock: minprice = min(minprice, price[i]) and profit = max(profit, price[i] - minprice)
22. alternate positive and negative:
```java
public static ArrayList<Integer> RearrangebySign(ArrayList<Integer> A) {
        int n = A.size();

        // Define array for storing the ans separately.
        ArrayList<Integer> ans = new ArrayList<>(Collections.nCopies(n, 0));

        // positive elements start from 0 and negative from 1.
        int posIndex = 0, negIndex = 1;
        for (int i = 0; i < n; i++) {

            // Fill negative elements in odd indices and inc by 2.
            if (A.get(i) < 0) {
                ans.set(negIndex, A.get(i));
                negIndex += 2;
            }

            // Fill positive elements in even indices and inc by 2.
            else {
                ans.set(posIndex, A.get(i));
                posIndex += 2;
            }
        }

        return ans;
    }
```

24. Calculate all permutations: 
```java
class Solution {
    public List<List<Integer>> permute(int[] nums) {
        List<List<Integer>> ans = new ArrayList<>();
        backtrack(nums, new ArrayList<>(), ans);
        return ans;
    }

    private void backtrack(int[] nums, List<Integer> path, List<List<Integer>> ans) {
        if (path.size() == nums.length) {
            ans.add(new ArrayList<>(path));
            return;
        }

        for (int n : nums) {
            if (!path.contains(n)) {
                path.add(n);
                backtrack(nums, path, ans);
                path.remove(path.size() - 1);
            }
        }
    }
	}
```
24. next permutation:
approach: find breaking point i.e find i where arr[i] is smaller than arr[i+1], if no breaking point exists then it means that array is sorted is descending order and no greater permutation exists. in this case, return the first permutation i.e. the reverse of array.
if breaking point exists then:
		1. swap breaking point and the right most greater element from breaking point.
		2. reverse array from breaking point

```java
public static List< Integer > nextGreaterPermutation(List< Integer > A) {
        int n = A.size(); // size of the array.
        // Step 1: Find the break point:
        int ind = -1; // break point
        for (int i = n - 2; i >= 0; i--) {
            if (A.get(i) < A.get(i + 1)) {
                // index i is the break point
                ind = i;
                break;
            }
        }
        // If break point does not exist:
        if (ind == -1) {
            // reverse the whole array:
            Collections.reverse(A);
            return A;
        }
        // Step 2: Find the next greater element and swap it with arr[ind]:
        for (int i = n - 1; i > ind; i--) {
            if (A.get(i) > A.get(ind)) {
                int tmp = A.get(i);
                A.set(i, A.get(ind));
                A.set(ind, tmp);
                break;
            }
        }
        // Step 3: reverse the right half:
        List<Integer> sublist = A.subList(ind + 1, n);
        Collections.reverse(sublist);
        return A;
    }
```

26. set matrix zero:
```java
class Solution {
    public void setZeroes(int[][] matrix) {
        int rows = matrix.length;
        int columns = matrix[0].length;
        boolean[] setRows = new boolean[rows];
        boolean[] setColumns = new boolean[columns];

        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < columns; j++) {
                if (matrix[i][j] == 0) {
                    setRows[i] = true;
                    setColumns[j] = true;
                }
            }
        }
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < columns; j++) {
                if (setRows[i] || setColumns[j]) {
                    matrix[i][j] = 0;
                }
            }
        }
    }
 }
```

27. Longest sequence:
```java
class Solution {
    public int longestConsecutive(int[] nums) {
        if(nums.length == 0) return 0;
        HashSet<Integer> set = new HashSet<>();
        for(int num : nums){
            set.add(num);
        }
        int maxCount = 1;
        for(int num : set){
            if(!set.contains(num-1)){
                int x = num;
                int count = 1;
                while(set.contains(x+1)){
                    count++;
                    x = x+1;
                }
                maxCount = Math.max(maxCount, count);
            }
        }
        return maxCount;
    }
 }
```

28. rotate square matrix by 90 degrees: **approach**: transpose matrix then reverse each row:
```
def rotate(self, matrix: List[List[int]]) -> None:
    # transpose matrix:
    n = len(matrix)
    for i in range(n):
        for j in range(i):
            matrix[i][j], matrix [j][i] = matrix [j][i], matrix [i][j]
        #reverse each row:
    for i in range(n):
        matrix[i].reverse()
```

28. Spiral traversal of matrix: use four loops
```
class Solution:
    def spiralOrder(self, matrix: List[List[int]]) -> List[int]:
        row = len(matrix)
        col = len(matrix[0])
        if row == 1: return matrix[0]
        read = []
        top = 0
        left = 0
        right = col - 1
        bottom = row - 1
        while top <= bottom and left <= right:
            for i in range(left, right + 1):
                read.append(matrix[top][i])
            top+=1
            for i in range(top, bottom + 1):
                read.append(matrix[i][right])
            right -= 1
            if (top <= bottom):
                for i in range(right, left-1, -1):
                    read.append(matrix[bottom][i])
            bottom -= 1
            if left<=right:
                for i in range(bottom, top -1, -1):
                    read.append(matrix[i][left])
            left += 1

        return read
```

29. count subarray sum:
		approach -> brute force use two loops but time limit exceeds so optimally:
			maintain a map of prefix sums i.e. basically sum of all numbers in array uptil index i, and store the count of the occurences of the prefix_sum, traverse the array and check if the current prefix_sum - target exists in the map, if it does then it means that you can make a subarray with target sum by removing the elements that gave sum = target + prefix_sum.
```
def subarraySum(self, nums: List[int], k: int) -> int:
        n = len(nums)
        array_count = 0
        pref_sum = 0
        prefix_map = {}
        prefix_map[0] = 1
        # this is important because if the target element itself exists in the array then it detects it as a valid subarray too only if we have 0 in map
        for i in range(n):
            pref_sum += nums[i]
            diff = pref_sum - k
            if diff in prefix_map:
                array_count += prefix_map[diff]
            if pref_sum in prefix_map:
                prefix_map[pref_sum] += 1
            else:
                prefix_map[pref_sum] = 1
        return array_count
```

30. **Pascal's triangle**: value at index [r]\[c] is equal to (r-1)C(c-1) combination
nCr = n! / r! x (n-r)!

**shortcut for calculating nCr in code: pattern to note is that: n will go three places and r will go same r places always**
![[Pasted image 20250721221437.png]]

```
class Solution:
    def nCr(self, n: int, r: int) -> int:
        res = 1

        for i in range(r):
            res *= (n-i)
            res //= (i+1)

        return res
    def generate(self, numRows: int) -> List[List[int]]:
        res = []
        for i in range(1, numRows+1):
            arr = []
            for j in range(1, i+1):
                arr.append(self.nCr(i-1, j-1))
            res.append(arr)
        return res
```

31. Majority Element II: use Extended Boyer Moore’s Voting Algorithm
	1. If **cnt1** is 0 and the current element is not el2 then store the current element of the array as **el1 along with increasing the cnt1 value by 1**.
	2. If **cnt2** is 0 and the current element is not el1 then store the current element of the array as **el2 along with increasing the cnt2 value by 1**.
	3. If the current element and **el1** are the same increase the **cnt1** by 1.
	4. If the current element and **el2** are the same increase the **cnt2** by 1.
	5. Other than all the above cases: decrease cnt1 and cnt2 by 1.
![[Pasted image 20250722194728.png]]
```
class Solution:
    def majorityElement(self, nums: List[int]) -> List[int]:
        cnt1, cnt2 = 0, 0
        el1, el2 = float('-inf'), float('-inf')
        res = []

        for num in nums:
            if num == el1:
                cnt1 += 1
            elif num == el2:
                cnt2 += 1
            elif cnt1 == 0:
                el1 = num
                cnt1 = 1
            elif cnt2 == 0:
                el2 = num
                cnt2 = 1
            else:
                cnt1 -= 1
                cnt2 -= 1
        
        cnt1 = 0
        cnt2 = 0

        for num in nums:
            if num == el1:
                cnt1 += 1
            if num == el2:
                cnt2 += 1

        mini = int(len(nums)/3) + 1
        if cnt1 >= mini:
            res.append(el1)
        if cnt2 >= mini:
            res.append(el2)

        return res
```

32. Merge intervals:
```java
public static List<List<Integer>> mergeOverlappingIntervals(int[][] arr) {
        int n = arr.length; // size of the array
        //sort the given intervals:
        /*Arrays.sort(arr, new Comparator<int[]>() {
            public int compare(int[] a, int[] b) {
                return a[0] - b[0];
            }
        }); */

        List<List<Integer>> ans = new ArrayList<>();

        for (int i = 0; i < n; i++) { // select an interval:
            int start = arr[i][0];
            int end = arr[i][1];

            //Skip all the merged intervals:
            if (!ans.isEmpty() && end <= ans.get(ans.size() - 1).get(1)) {
                continue;
            }

            //check the rest of the intervals:
            for (int j = i + 1; j < n; j++) {
                if (arr[j][0] <= end) {
                    end = Math.max(end, arr[j][1]);
                } else {
                    break;
                }
            }
            ans.add(Arrays.asList(start, end));
        }
        return ans;
    }
```

#### Strings:
32. Remove outer parentheses: 
```
def removeOuterParentheses(self, s: str) -> str:
    depth=0
    res=""
    for ch in s:
        if ch=="(":
            if depth>0:
                res+=ch
            depth+=1
        elif ch==")":
            depth-=1
            if depth>0:
                res+=ch
    return res
```

33. **Compress String:**
```java
class Solution {
    public int compress(char[] chars) {
        int read = 0, write = 0;
        int n = chars.length;

        while(read < n){
            int count = 0;
            char currentChar = chars[read];

            while (read < n && currentChar == chars[read]){
                read++;
                count++;
            }

            chars[write++] = currentChar;

            if(count > 1){
                for(char c : Integer.toString(count).toCharArray()){
                    chars[write++] = c;
                }
            }

        }

        return write;
    }
 }
```

34. Largest odd number in sring:
```
class Solution:
    def largestOddNumber(self, num: str) -> str:
        largest_num=""
        for i in range(len(num)-1,-1,-1):
            if int(num[i])%2!=0:
               largest_num=num[0:i+1]
               return largest_num
        return largest_num
```

35. Longest common prefix:
```
class Solution:
    def longestCommonPrefix(self, strs: List[str]) -> str:
        prefix = strs[0]
        for word in strs[1:]:
            i = 0
            while i < len(word) and i < len(prefix) and prefix[i] == word[i]:
                i+=1
            prefix = prefix[:i]
            if not prefix:
                return ""
        return prefix
```

36. Isomorphic strings:
```
class Solution {
    public boolean isIsomorphic(String s, String t) {
        Map<Character, Character> sMap = new HashMap<>();
        Map<Character, Character> tMap = new HashMap<>();

        for(int i = 0; i < s.length(); i++){
            char chS = s.charAt(i);
            char chT = t.charAt(i);

            if(sMap.containsKey(chS)){
                if(sMap.get(chS) != chT)
                    return false;
            }
            else{
                sMap.put(chS, chT);
            }

            if(tMap.containsKey(chT)){
                if(tMap.get(chT) != chS){
                    return false;
                }
            }
            else{
                tMap.put(chT, chS);
            }
        }
        return true;
    }
 }
```

37. Reverse words in a string:
```java
class Solution {
    public String reverseWords(String s) {
        s = s.trim();
        String[] words = s.split("\\s+");
        StringBuilder result = new StringBuilder("");

        for(int i = words.length - 1; i>=0; i--){
            result.append(words[i]);
            if(i!=0)
                result.append(" ");
        }

        return result.toString();
    }
 }
```

38. Rotate String:
```
class Solution {
    public boolean rotateString(String s, String goal) {
        if(s.length() != goal.length()) return false;
        String ss = s + s;
        return ss.contains(goal);
    }
 }
```

39. 3 Sum:
```
class Solution {
    public List<List<Integer>> threeSum(int[] nums) {
        Arrays.sort(nums);
        List<List<Integer>> res = new ArrayList<>();
        int n = nums.length;
        for(int i = 0; i < n; i++){
            if(i!= 0 && nums[i] == nums[i-1]) continue;

            int j = i + 1;
            int k = n - 1;

            while(j < k){
                int sum = nums[i] + nums[j] + nums[k];
                if(sum > 0) k--;
                else if(sum < 0) j++;
                else{
                    List<Integer> temp = Arrays.asList(nums[i], nums[j], nums[k]);
                    res.add(temp);
                    j++;
                    k--;
                    while(j < k && nums[j] == nums[j-1]) j++;
                    while(k < n-1 && j<k && nums[k] == nums[k+1]) k--;
                }
            }   
        }
        return res;
    }
}
```

38. Binary Search
39. Find first and last occurrence:
```
class Solution {
    public int[] searchRange(int[] nums, int target) {
        if(nums.length == 0) return new int[]{-1, -1};
        int first = findBound(nums, target, true);
        if(first == -1) return new int[] { -1, -1};
        int last = findBound(nums, target, false);
        return new int[]{first, last};
    }

    public int findBound(int[] nums, int target, boolean pos){
        int low = 0, high = nums.length - 1;
        int res = -1;

        while(low <= high){
            int mid = (low+high) / 2;
            if(nums[mid] == target){
                res = mid;
                if(pos){
                    high = mid-1;
                }
                else{
                    low = mid + 1;
                }
            }
            else if(nums[mid] > target) high = mid - 1;
            else low = mid + 1;
        }
        return res;
    }
}
```

40. Find index of element in rotated array:
```
class Solution {
    public int search(int[] nums, int target) {
        int low = 0, high = nums.length - 1;
        while(low <= high){
            int mid = (low+high)/2;
            if(nums[mid] == target)
            return mid;
            if(nums[low] <= nums[mid]){
                if(nums[mid] >= target && nums[low] <= target)
                high = mid - 1;
                else
                low = mid+1;
            }
            else{
                if(nums[high] >= target && nums[mid] <= target)
                low = mid + 1;
                else
                high = mid - 1;
            }
        }
        return -1;
    }
}
```

41. Search in rotated array II:
```
class Solution {
    public boolean search(int[] nums, int target) {
        int low = 0, high = nums.length - 1;
        while(low <= high){
            int mid = (low+high)/2;
            if(nums[mid] == target)
            return true;

            if(nums[low] == nums[mid] && nums[mid] == nums[high]){
                low = low+1;
                high = high - 1;
                continue;
            }

            if(nums[low]<=nums[mid]){
                if(nums[mid] >= target && nums[low] <= target)
                high = mid-1;
                else
                low = mid + 1;
            }
            else{
                if(nums[mid] <= target && target <= nums[high])
                low = mid + 1;
                else
                high = mid - 1;
            }
        }
        return false;
    }
}
```

42. Find single element: nice logic
```
class Solution {
    public int singleNonDuplicate(int[] nums) {
        int n = nums.length;
        if(n == 1) return nums[0];
        if(nums[0] != nums[1]) return nums[0];
        if(nums[n-1] != nums[n-2]) return nums[n-1];
        int low = 1, high = n - 2;
        while(low <= high){
            int mid = (low+high)/2;
            if(nums[mid] != nums[mid+1] && nums[mid] != nums[mid-1]) return nums[mid];

            if((mid%2 == 1 && nums[mid]==nums[mid-1]) || (mid%2==0 && nums[mid]==nums[mid+1]))
            low = mid + 1;
            else
            high = mid - 1;
        }
        return -1;
    }
}
```

43. Find peak:
```
class Solution {
    public int findPeakElement(int[] nums) {
        int n = nums.length;
        if(n==1) return 0;
        if(nums[0] > nums[1]) return 0;
        if(nums[n-1]>nums[n-2]) return n-1;
        int low = 1, high = n - 2;
        while(low <= high){
            int mid = (low+high)/2;
            if(nums[mid-1] < nums[mid] && nums[mid+1] < nums[mid])
            return mid;
            if(nums[mid] > nums[mid-1]) low = mid+1;
            else high = mid - 1;
        }
        return -1;
    }
}
```

#### Linked Lists
44. Delete node without having access to head node:
```
class Solution {
    public void deleteNode(ListNode node) {
        while(node.next.next!=null){
            node.val = node.next.val;
            node = node.next;
        }
        node.val = node.next.val;
        node.next = null;
    }
}
```

45. Find loop starting in LL OR LinkedList Cycle II:
```
public class Solution {
    public ListNode detectCycle(ListNode head) {
        ListNode hare = head, tortoise = head;
        while(hare!=null && hare.next!=null){
            hare = hare.next.next;
            tortoise = tortoise.next;
            if(hare == tortoise){
                tortoise = head;
                while(tortoise != hare){
                    tortoise = tortoise.next;
                    hare = hare.next;
                }
                return tortoise;
            }
        }
        return null;
    }
}
```

46. LL Palindrome:
```
class Solution {
    public boolean isPalindrome(ListNode head) {
        if (head == null || head.next == null) return true;
        ListNode slow = head, fast = head;
        while (fast != null && fast.next != null) {
            slow = slow.next;
            fast = fast.next.next;
        }
        ListNode prev = null, curr = slow;
        while (curr != null) {
            ListNode next = curr.next;
            curr.next = prev;
            prev = curr;
            curr = next;
        }
        while (prev != null) {
            if (head.val != prev.val) return false;
            head = head.next;
            prev = prev.next;
        }
        return true;
    }
}
```

47. Odd Even LinkedList:
```
class Solution {
    public ListNode oddEvenList(ListNode head) {
        ListNode oddHead = new ListNode(-1), odd = oddHead;
        ListNode evenHead = new ListNode(-1), even = evenHead;
        ListNode curr = head, temp;
        boolean isOdd = true;
        while(curr!=null){
            //broke the link between continuous odd-even nodes:
            temp = curr;
            curr = curr.next;
            temp.next = null;
            if(isOdd){
                odd.next = temp;
                odd = odd.next;
            }
            else{
                even.next = temp;
                even = even.next;
            }
            isOdd = !isOdd;
        }
        odd.next = evenHead.next;
        return oddHead.next;
    }
}
```

48. Delete n node from last:
```
class Solution {
    public ListNode removeNthFromEnd(ListNode head, int n) {
        ListNode slow = head, fast = head;
        for(int i = 0; i < n; i++){
            fast = fast.next;
        }
        //important edge case when n is equal to length of list, we need to delete first node.
        if(fast == null) return head.next;
        while(fast.next!=null){
            slow = slow.next;
            fast = fast.next;
        }
        slow.next = slow.next.next;
        return head;
    }
}
```

49. Delete middle node:
```
class Solution {
    public ListNode deleteMiddle(ListNode head) {
        if(head == null || head.next == null) return null;
        ListNode slow = head, fast = head.next.next;
        while(fast!=null && fast.next!=null){
            slow = slow.next;
            fast = fast.next.next;
        }
        ListNode del = slow.next;
        slow.next = slow.next.next;
        del = null;
        return head;
    }
}
```

50. Merge sort on LL:
```
class Solution {
    public ListNode sortList(ListNode head) {
        //base and edge case:
        if(head == null || head.next == null){
            return head;
        }
        ListNode mid = middle(head);
        ListNode right = mid.next;
        mid.next = null;
        ListNode left = head;

        left = sortList(left);
        right = sortList(right);

        return merge(left, right);
    }

    public ListNode merge(ListNode left, ListNode right){
        ListNode dummy = new ListNode(-1), temp = dummy;

        while(left!=null && right!=null){
            if(left.val <= right.val){
                temp.next = left;
                left = left.next;
            }
            else{
                temp.next = right;
                right = right.next;
            }
            temp = temp.next;
        }
        if(left!=null){
            temp.next = left;
        }
        if(right!=null){
            temp.next = right;
        }
        return dummy.next;
    }

    public ListNode middle(ListNode head){
        if(head == null || head.next == null) return head;
        ListNode slow = head, fast = head.next;
        while(fast!=null && fast.next!=null){
            fast = fast.next.next;
            slow = slow.next;
        }
        return slow;
    }
}
```

51. Dutch national flag using LL:
```
class Solution {
    public ListNode sortList(ListNode head) {
        int[] count = new int[3];
        ListNode temp = head;
        while(temp!=null){
            count[temp.val]++;
            temp = temp.next;
        }
        int i = 0;
        temp = head;
        while(temp!=null){
            if(count[i]==0) i++;
            else{
                temp.val = i;
                temp = temp.next;
                count[i]--;
            }
        }
        return head;
    }
}
```

52. Intersection of lists:
```
public class Solution {
    public ListNode getIntersectionNode(ListNode headA, ListNode headB) {
        int diff = difference(headA, headB);

        if(diff > 0){
            for(int i = 0; i < diff; i++){
                headA = headA.next;
            }
        }
        else if(diff < 0){
            for(int i = 0; i < -diff; i++) headB = headB.next;
        }

        while(headA!=null){
            if(headA == headB) return headA;
            headA = headA.next;
            headB = headB.next;
        }
        return null;
    }

    public int difference(ListNode a, ListNode b){
        int lenA = 0, lenB = 0;
        while(a!=null){
            lenA++;
            a = a.next;
        }
        while(b!=null ){
            lenB++;
            b = b.next;
        }
        return lenA - lenB;
    }
}
```

53. Add one to value represented by LL (not on leetcode):
```
class Solution {
    public ListNode addOne(ListNode head) {
        // Step 1: Reverse the list
        head = reverse(head);

        // Step 2: Add 1 to the reversed list
        ListNode temp = head;
        int carry = 1;

        while (temp != null) {
            int sum = temp.val + carry;
            temp.val = sum % 10;
            carry = sum / 10;

            if (temp.next == null && carry != 0) {
                temp.next = new ListNode(0); // extend the list if needed
            }

            temp = temp.next;
        }

        // Step 3: Reverse the list again
        head = reverse(head);
        return head;
    }

    // Helper function to reverse a linked list
    private ListNode reverse(ListNode head) {
        ListNode prev = null;
        ListNode curr = head;

        while (curr != null) {
            ListNode nextNode = curr.next;
            curr.next = prev;
            prev = curr;
            curr = nextNode;
        }

        return prev;
    }
}
```

54. Add two numbers in LL:
```
class Solution {
    public ListNode addTwoNumbers(ListNode l1, ListNode l2) {
        if(l1==null) return l2;
        if(l2==null) return l1;
        int carry = 0;
        ListNode num1 = l1, num2 = l2, result = new ListNode(-1), temp = result;
        while(num1!=null && num2!=null){
            int sum = num1.val + num2.val + carry;
            ListNode newNode = new ListNode(sum % 10);
            carry = sum / 10;
            num1 = num1.next;
            num2 = num2.next;
            temp.next = newNode;
            temp = temp.next;
        }

        while(num1!=null){
            ListNode newNode = new ListNode((num1.val + carry) % 10);
            carry = (num1.val + carry)/10;
            temp.next = newNode;
            temp = temp.next;
            num1 = num1.next;
        }

        while(num2!=null){
            ListNode newNode = new ListNode((num2.val + carry) % 10);
            carry = (num2.val + carry)/10;
            temp.next = newNode;
            temp = temp.next;
            num2 = num2.next;
        }

        if(num1==null && num2==null && carry != 0){
            ListNode newNode = new ListNode(carry);
            temp.next = newNode;
        }

        return result.next;
    }
}
```

55. Rotate LL:
```
class Solution {
    public ListNode rotateRight(ListNode head, int k) {
        if(head == null || head.next == null || k==0) return head;
        ListNode temp = head;
        int len = 1;
        while(temp.next!=null){
            ++len;
            temp = temp.next;
        }
        temp.next = head;
        k=k%len;
        int end = len - k;
        while(end--!=0){
            temp = temp.next;
        }
        head = temp.next;
        temp.next = null;
        return head;
    }
}
```

56. Create deep copy of linked list:
    approach 1: use hashmap to map original node to copy node
    approach 2: insert a new node after every node in original list and update random pointer in another loop using newNode.random = temp.random.next
```
class Solution {
    public Node copyRandomList(Node head) {
        if(head == null) return null;
        Node temp = head;
        while(temp!=null){
            Node newNode = new Node(temp.val);
            newNode.next = temp.next;
            temp.next = newNode;
            temp = newNode.next;
        }
        temp = head;
        while(temp!=null){
            Node newNode = temp.next;
            if(temp.random != null){
                newNode.random = temp.random.next;
            }
            temp = temp.next.next;
        }
        Node newHead = head.next;
        Node newTemp = newHead;
        temp = head;
        while(temp!=null){
            temp.next = temp.next.next;
            if(newTemp.next!=null) newTemp.next = newTemp.next.next;
            temp = temp.next;
            newTemp = newTemp.next;
        }
        return newHead;
    }
}
```

#### Recursion and backtracking:
**DRAW A TREE AND EACH BRANCH SHOULD BE LIKE "DO THIS" and "DONT DO THIS"

57. pow(x,y): look for solution in basics
58. count Good numbers: note: mod is there only to ensure the ans can fit inside "int" datatype, it was given in question.
```
class Solution {
    public int countGoodNumbers(long n) {
        if(n==1) return 5;
        long even = (n+1)/2;
        long odd = n/2;
        long mod = 1000000000 + 7;
        return (int)((helper(5, even, mod)*helper(4, odd, mod))%mod);
    }
    public long helper(long x, long n, long mod){
        long ans = 1;
        while(n > 0){
            if(n%2 == 0){
                x = (x*x)%mod;
                n/=2;
            }
            else{
                ans=(x*ans)%mod;
                n--;
            }
        }
        return ans;
    }
}
```

59. Reverse stack using recursion: refer to basics
60. Generate parentheses:
```
class Solution {
    public List<String> generateParenthesis(int n) {
        List<String> result = new ArrayList<>();
        if(n <= 0) return result;
        int open = 0, close = 0;
        recursive(n, open,close,"", result);
        return result;
    }

    public void recursive(int n, int open, int close, String curr, List<String> result){
        if(open == n && close == n){
            result.add(curr);
            return;
        }
        if(open < n){
            recursive(n, open+1, close, curr+"(", result);
        }
        if(close < open){
            recursive(n, open, close+1, curr+")", result);
        }

    }
}
```

61. Subsets:
```
class Solution {
    public List<List<Integer>> subsets(int[] nums) {
        List<List<Integer>> result = new ArrayList<>();
        List<Integer> curr = new ArrayList<>();
        int index = 0;
        backtrack(index, nums, curr, result);
        return result;
    }

    public void backtrack(int index, int[] nums, List<Integer> curr, List<List<Integer>> result){
        if(index == nums.length){
            result.add(new ArrayList<>(curr));
            return;
        }
        // you have two choices, either exclude element at index or include elem at index
        backtrack(index + 1, nums, curr, result);
        curr.add(nums[index]);
        backtrack(index+1, nums, curr, result);
        //backtrack from choosing element in curr for all possibilities:
        curr.remove(curr.size()-1);
    }
}
```

62. Check if subsequence with sum k exists:
```
class Solution {
    public boolean checkSubsequenceSum(int[] nums, int k) {
        int n = nums.length;
        int index = 0;
        int sum = 0;
        return func(index, sum, k, nums, n);
    }
    
    public boolean func(int index, int sum, int k, int[] nums, int n) {
        if (index == n) {
            if (sum == k) return true;
            return false;
        }
        
        // not take:
        boolean notTake = func(index + 1, sum, k, nums, n);
        
        // take:
        boolean take = func(index + 1, sum + nums[index], k, nums, n);
        
        // Return true if either choice leads to a solution
        return notTake || take;
    }
}
```

63. Combination sum:
```
class Solution {
    public List<List<Integer>> combinationSum(int[] candidates, int target) {
        List<List<Integer>> result = new ArrayList<>();
        List<Integer> chosen = new ArrayList<>();
        int n = candidates.length;
        func(0, chosen, 0, candidates, target, result);
        return result;
    }

    public void func(int index, List<Integer> chosen, int sum, int[] cand, int target, List<List<Integer>> result){
        if(sum == target){
            result.add(new ArrayList<>(chosen));
            return;    
        }
        if(index == cand.length || sum > target){
            return;
        }
        //take element:
        int temp = cand[index]; 
        chosen.add(temp);
        func(index, chosen, sum+temp, cand, target, result);
        chosen.remove(chosen.size()-1);
        //not take this element:
        func(index+1, chosen, sum, cand, target, result);
    }
}
```

64. Combination sum 2:
1: 
```
class Solution {
    public List<List<Integer>> combinationSum2(int[] candidates, int target) {
        List<List<Integer>> result = new ArrayList<>();
        List<Integer> chosen = new ArrayList<>();
        int n = candidates.length;
        Arrays.sort(candidates);
        func(0, chosen, 0, candidates, target, result);
        return result;
    }

    public void func(int index, List<Integer> chosen, int sum, int[] cand, int target, List<List<Integer>> result){
        if(sum == target){
            result.add(new ArrayList<>(chosen));
            return;    
        }
        if(index == cand.length || sum > target){
            return;
        }
        //take element:
        int temp = cand[index];
        chosen.add(temp);
        func(index+1, chosen, sum+temp, cand, target, result);
        chosen.remove(chosen.size()-1);
        //not take this element:
        while(index+1 < cand.length && cand[index] == cand[index+1]){
            index++;
        }
        func(index+1, chosen, sum, cand, target, result);
    }
}
```

2:
```
class Solution {
    public List<List<Integer>> combinationSum2(int[] candidates, int target) {
        Arrays.sort(candidates);
        List<List<Integer>> res = new ArrayList<>();
        List<Integer> chosen = new ArrayList<>();
        func(0, chosen, res, candidates, target);
        return res;
    }

    public void func(int index, List<Integer> chosen, List<List<Integer>> result, int[] cand, int k){
        if(k == 0){
            result.add(new ArrayList<>(chosen));
            return;
        }
        for(int i = index; i < cand.length; i++){
            if(i > index && cand[i] == cand[i-1]) continue; //ensures no duplicates
            if(cand[index] > k) break; //pruning

            chosen.add(cand[i]);
            func(i+1, chosen, result, cand, k-cand[i]);
            chosen.remove(chosen.size()-1);
        }
    }
}
```

64. Subset Sum:
```
class Solution {
    public List<Integer> subsetSums(int[] nums) {
      List<Integer> result = new ArrayList<>();
      subsets(0, 0, nums, result);
      return result;
    }

    public void subsets(int index, int sum, int[] nums, List<Integer> result){
      if(index == nums.length){
        result.add(sum);
        return;
      }

      //take element:
      subsets(index+1, sum+nums[index], nums, result);
      // not take:
      subsets(index+1, sum, nums, result);
    }
}
```

65. Subset sum II:
```
class Solution {
    public List<List<Integer>> subsetsWithDup(int[] nums) {
        Arrays.sort(nums);
        List<List<Integer>> result = new ArrayList<>();
        List<Integer> st = new ArrayList<>();
        subsets(0, st, nums, result);
        return result;
    }

    public void subsets(int index, List<Integer> st, int[] nums, List<List<Integer>> result){
        if(index == nums.length){
            result.add(new ArrayList<>(st));
            return;
        }
        //take:
        st.add(nums[index]);
        subsets(index+1,st , nums, result);
        st.remove(st.size()-1);
        //not take:
        while(index+1 < nums.length && nums[index] == nums[index+1]) index++;
        subsets(index+1, st, nums, result);
    }
}
```

66. Combination sum III:
my approach:
```
class Solution {
    public List<List<Integer>> combinationSum3(int k, int n) {
        List<List<Integer>> result = new ArrayList<>();
        List<Integer> chosen = new ArrayList<>();
        if(n < k) return result;
        int[] nums = new int[] {1,2,3,4,5,6,7,8,9};
        getCombinations(0, 0, chosen, nums, k, n, result);
        return result;
    }

    public void getCombinations(int index, int sum,List<Integer> chosen, int[] nums, int k, int target, List<List<Integer>> result){
        //base case:
        if(chosen.size() == k){
            if(sum == target) result.add(new ArrayList<>(chosen));
        return;
        }
        if(index == nums.length) return;
        //take this element:
        chosen.add(nums[index]);
        getCombinations(index+1, sum+nums[index], chosen, nums, k, target, result);
        chosen.remove(chosen.size()-1);
        //not take:
        getCombinations(index+1, sum, chosen, nums, k, target, result);
    }
}
```

further optimised:
```
class OptimizedSolution1 {
    public List<List<Integer>> combinationSum3(int k, int n) {
        List<List<Integer>> result = new ArrayList<>();
        
        // Early termination checks
        if (k > 9 || n < k || n > 45) return result; // 45 = 1+2+...+9
        
        backtrack(1, k, n, new ArrayList<>(), result);
        return result;
    }
    
    private void backtrack(int start, int k, int target, 
                          List<Integer> current, List<List<Integer>> result) {
        // Early pruning: if target is too small or we need too many numbers
        if (target < 0 || current.size() > k) return;
        
        // Success case
        if (current.size() == k && target == 0) {
            result.add(new ArrayList<>(current));
            return;
        }
        
        // Try numbers from start to 9
        for (int i = start; i <= 9; i++) {
            // More pruning: if current number is larger than remaining target
            if (i > target) break;
            
            current.add(i);
            backtrack(i + 1, k, target - i, current, result);
            current.remove(current.size() - 1);
        }
    }
}
```

67. Letter combinations of a phone number:
```
class Solution {
    public List<String> letterCombinations(String digits) {
        List<String> result = new ArrayList<>();
        if(digits.length() == 0) return result;
        String[] map = {
            "abc",
            "def",
            "ghi",
            "jkl",
            "mno",
            "pqrs",
            "tuv",
            "wxyz"
        };
        StringBuilder chosen = new StringBuilder();
        combinations(0, chosen, digits, map, result);
        return result;
    }

    public void combinations(int index, StringBuilder chosen, String digits, String[] map, List<String> result){
        if(index == digits.length()){
            result.add(chosen.toString());
            return;
        }
        int digit = digits.charAt(index) - '2';
        String letters = map[digit];
        for(char letter : letters.toCharArray()){
            chosen.append(letter);
            combinations(index+1, chosen, digits, map, result);
            chosen.deleteCharAt(chosen.length()-1);
        }
    }
}
```

68. Palindrome Partitioning:
```
class Solution {
    public List<List<String>> partition(String s) {
        List<List<String>> result = new ArrayList<>();
        List<String> chosen = new ArrayList<>();
        substring(0, s, chosen, result);
        return result;
    }

    public void substring(int index, String s, List<String> chosen, List<List<String>> result){
        if(index == s.length()){
            result.add(new ArrayList<>(chosen));
            return;
        }

        for(int i = index; i < s.length(); i++){
            if(isPalindrome(s, index, i)){
                chosen.add(s.substring(index, i+1));
                substring(i+1, s, chosen, result);
                chosen.remove(chosen.size()-1);
            }
        }
    }

    public boolean isPalindrome(String s, int start, int end){
        while(start <= end){
            if(s.charAt(start++)!=s.charAt(end--)) return false;
        }
        return true;
    }
}
```

##### Bit Manipulation:
69. check power of 2:
```
class Solution {
	public:
	    bool isPowerOfTwo(int n) {
	        if( n <= 0)
	        return false;
	
	        bool isPower = (n & (n-1)) == 0 ;
	
	        return isPower;
	    }
};
```

##### Stack
70. trapping rain water:
```java
class Solution {
    public int trap(int[] height) {
        // if num > stack.peek() -> pop last and push current
        Stack<Integer> stack = new Stack<>();
        int count = 0;
        for(int i = 0; i < height.length ; i++){
            while(!stack.isEmpty() && height[i] > height[stack.peek()]){
                int bottom = stack.pop();
                if(stack.isEmpty()) break;
                int right = i;
                int left = stack.peek();
                int width = right - left - 1;
                int water = Math.min(height[left], height[right]) - height[bottom];
                count += water*width;
            }
            stack.push(i);
        }
        return count;
    }
}
```

71. reverse k groups:
    ```
    public static ListNode reverseKGroup(ListNode head, int k) {
        if (head == null || k == 1) return head;
        
        // Check if we have k nodes remaining
        ListNode curr = head;
        int count = 0;
        while (curr != null && count < k) {
            curr = curr.next;
            count++;
        }
        
        // If we have k nodes, reverse them
        if (count == k) {
            curr = reverseKGroup(curr, k); // Recursively reverse rest
            
            // Reverse current k nodes
            ListNode prev = curr;
            ListNode current = head;
            
            while (count > 0) {
                ListNode next = current.next;
                current.next = prev;
                prev = current;
                current = next;
                count--;
            }
            head = prev;
        }
        return head;
    }
    ```
##### Sliding Window:
71. Longest substring without repeating chracters:
```java
class Solution {
    public int lengthOfLongestSubstring(String s) {
        HashMap<Character, Integer> seen = new HashMap<>();
        int left = 0, n = s.length(), count = 0;;
        for(int right = 0; right < n; right++){
            char temp = s.charAt(right);
            if(seen.containsKey(temp)){
                left = Math.max(left, seen.get(temp)+1);
            }
            seen.put(temp, right);
            count = Math.max(count, right-left+1);
        }
        return count;
    }
 }
```

72. Max consecutive ones III:
```java
class Solution {
    public int longestOnes(int[] nums, int k) {
        int zero = 0;
        int left = 0, right = 0;
        int res = 0;
        while(right < nums.length){
            if(nums[right] == 0){
                zero++;
            }
            right++;
            while(k < zero){
                if(nums[left] == 0){
                    zero--;
                }
                left++;
            }
            res = Math.max(res, right-left);
        }
        return res;
    }
 }
```

73. Longest common subsequence:
```java
class Solution {
    public int longestCommonSubsequence(String text1, String text2) {
        int m = text1.length(), n = text2.length();
        Map<String, Integer> seen = new HashMap<>();
        return lcs(0, 0, text1, text2, seen);
    }
    private int lcs(int i, int j, String text1, String text2, Map<String, Integer> memo) {
        // Base case
        if (i >= text1.length() || j >= text2.length()) {
            return 0;
        }
        // Fix 2: Create proper key for memoization
        String key = i + "," + j;
        if (memo.containsKey(key)) {
            return memo.get(key);
        }
        int result;
        if (text1.charAt(i) == text2.charAt(j)) {
            result = 1 + lcs(i + 1, j + 1, text1, text2, memo);
        } else {
            result = Math.max(
                    lcs(i, j + 1, text1, text2, memo), // Skip char from text1
                    lcs(i + 1, j, text1, text2, memo) // Skip char from text2
            );
        }
        // Fix 3: Store result in memo before returning
        memo.put(key, result);
        return result;
    }
 }
```

74. Maximal Square:
```java
class Solution {
    public int maximalSquare(char[][] matrix) {
        int rows = matrix.length, cols = matrix[0].length;
        int maxSide = 0;
        int[][] dp = new int[rows][cols];
        for(int[] row: dp){
            Arrays.fill(row, 0);
        }
        for(int i = 0; i < rows; i++){
            for(int j = 0; j < cols; j++){
                if(matrix[i][j] == '1'){
                    if(i == 0 || j == 0){
                        dp[i][j] = 1;
                    }
                    else{
                        int min = Math.min(dp[i-1][j], dp[i][j-1]);
                        dp[i][j] = Math.min(min, dp[i-1][j-1]) + 1;
                    }
                }
                maxSide = Math.max(maxSide, dp[i][j]);
            }
        }
        return maxSide*maxSide;
    }
 }
```

##### Graphs:
1. BFS and DFS: check basics

##### Priority Queue:
1. Task Scheduler:
```java
class Solution{
    public int leastInterval(char[] tasks, int n) {
        HashMap<Character, Integer> map = new HashMap<>();
        for(char task : tasks){
            map.put(task, map.getOrDefault(task, 0)+1);
        }
        PriorityQueue<Integer> maxHeap = new PriorityQueue<>((a,b) -> b - a);
        int total = 0;
        maxHeap.addAll(map.values());
        while(!maxHeap.isEmpty()){
            int cycle = 0;
            List<Integer> temp = new ArrayList<>();
            // we use i<= n because "idle" also takes one interval
            for(int i = 0; i <= n; i++){
                if(!maxHeap.isEmpty()){
                    int freq = maxHeap.poll();
                    if(freq > 1) temp.add(freq-1);
                    cycle++;
                }
            }
            for(int freq : temp) maxHeap.offer(freq);
            total += maxHeap.isEmpty() ? cycle : (n+1);
            //if the maxHeap is not empty it means that we used
            //full cycle length so add n+1 (+1 for the "idle")
        }
        return total;
    }
 }
```

2. Hands of straight:
```java
class Solution {
    public boolean isNStraightHand(int[] hand, int k) {
        int n = hand.length;
        if(n%k != 0) return false;
        HashMap<Integer, Integer> map = new HashMap<>();
        for(int card: hand){
            map.put(card, map.getOrDefault(card, 0)+1);
        }
        PriorityQueue<Integer> minHeap = new PriorityQueue<>(map.keySet());
        while(!minHeap.isEmpty()){
            int card = minHeap.peek();
            for(int j = 0; j < k; j++){
                int curr = card+j;
                if(!map.containsKey(curr) || map.get(curr) == 0){
                    return false;
                }                
                map.put(curr, map.get(curr)-1);
                if(map.get(curr) == 0){
                    minHeap.poll();
                }
            }
        }
        return true;
    }
 }
```

3. Kth Largest Element:
```java
class KthLargest {
    PriorityQueue<Integer> minHeap;
    int k;
    public KthLargest(int k, int[] nums) {
        minHeap  = new PriorityQueue<>();
        this.k = k;
        for(int num : nums) this.add(num);
    }
    public int add(int val) {
        minHeap.offer(val);
        if(minHeap.size() > k) minHeap.poll();
        return minHeap.peek();
    }
 }
```

4. k most frequent elements:
```java
class Solution {
    public int[] topKFrequent(int[] nums, int k) {
        PriorityQueue<Map.Entry<Integer, Integer>> minHeap = new PriorityQueue<>((a, b) -> a.getValue() - b.getValue());
        Map<Integer, Integer> map = new HashMap<>();
        for (int num : nums) {
            map.put(num, map.getOrDefault(num, 0) + 1);
        }
        for (Map.Entry<Integer, Integer> num : map.entrySet()) {
            minHeap.add(num);
            if (minHeap.size() > k) {
                minHeap.poll();
            }
        }
        int[] res = new int[k];
        for (int i = 0; i < k; i++)
            res[i] = minHeap.poll().getKey();
        return res;
    }
 }
```

5. 