from typing import List
import numpy as np
import array as ar

"""
crap code for canBeIncreasing():
        ns = sorted(nums)
        count = 0
        v = len(nums)
        flag = False
        in_flag = 0
        for i in range(1, len(nums)):
            if nums[i - 1] < nums[i]:
                pass
            else:
                in_flag += 1
        if in_flag > 1:
            return False
        for i in range(1, len(ns)):

            if len(nums) == 2:
                return True
            for j in range(0, len(nums)):
                flag = False
                for i in range(1, len(nums)):
                    if j == i - 1:
                        continue
                    if nums[i - 1] < nums[i]:
                        if flag==False and i==len(nums)-1:
                            return True
                    else:
                        flag=True
                        return False
            if flag:
                return False

        return True
"""


class Solution:
    def shuffle_array(self, nums: List[int], n: int) -> List[int]:
        """
        Best solution: 48 Ms
        def shuffle(self, nums: List[int], n: int) -> List[int]:
        ans = []
        for i in range(n):
            ans.append(nums[i])
            ans.append(nums[i+n])
        return ans

        Input: nums = [2, 5, 1, 3, 4, 7], n = 3
        Output: [2, 3, 5, 4, 1, 7]
        Explanation: Since
        x1 = 2, x2 = 5, x3 = 1, y1 = 3, y2 = 4, y3 = 7
        then
        the
        answer is [2, 3, 5, 4, 1, 7].
        O(n) space
        print("for O(N) space")
        res = []
        ptr1 = 0
        ptr2 = n
        for i in range(0, 2 * n, 2):
            print("ptr1 :", ptr1, "ptr2 :", ptr2)
            res.append(nums[ptr1])
            ptr1 += 1
            res.append(nums[ptr2])
            ptr2 += 1
        print(res)
        print("for O(1) space")

        3/2=1 4/2=2
        Input: nums = [2,5,1,3,4,7], n = 3
        Output: [2,3,5,4,1,7]
        Explanation: Since x1=2, x2=5, x3=1, y1=3, y2=4, y3=7 then the answer is [2,3,5,4,1,7].
        """
        q = 1001
        for i in range(0, n):
            r = nums[i]
            b = nums[n + i] % q
            nums[i] = q * b + r
        k = 2 * n - 1
        for i in range(n - 1, -1, -1):
            nums[k] = nums[i] // q
            nums[k - 1] = nums[i] % q
            k -= 2
        print(nums)

    def kidsWithCandies(self, candies: List[int], extraCandies: int) -> List[bool]:
        """
        Input: candies = [2,3,5,1,3], extraCandies = 3
        Output: [true,true,true,false,true]
        :param candies:
        :param extraCandies:
        :return:
        """
        max_candies = max(candies)
        res = [False for i in range(len(candies))]
        for i in range(len(candies)):
            if candies[i] + extraCandies >= max_candies:
                res[i] = True
        print(res)
        print("Best Solution MS")
        max_candies = max(candies)
        for i in range(len(candies)):
            candies[i] = candies[i] + extraCandies >= max_candies
        return candies

    def buildArray(self, nums: List[int]) -> List[int]:
        q = len(nums)
        for i in range(0, q):
            r = nums[i]
            b = nums[r] % q
            nums[i] = q * b + r

        for i in range(0, q):
            nums[i] //= q
        return nums

    def smallerNumbersThanCurrent(self, nums: List[int]) -> List[int]:
        count = [0] * 101
        print(count)
        ans = [0] * len(nums)
        for i in range(0, len(nums)):
            count[nums[i]] += 1
        print(count)
        for i in range(1, 100):
            count[i] += count[i - 1]
        print(count)
        for i in range(0, len(nums)):
            if nums[i] == 0:
                ans[i] = 0
            else:
                ans[i] = count[nums[i] - 1]
        return ans

    def restoreString(self, s: str, indices: List[int]) -> str:
        st = list(s)
        for i in range(0, len(s)):
            st[indices[i]] = s[i]
        return st

    def debug_restoreString(self, s: str, indices: List[int]) -> str:
        result = ''
        for i in range(len(s)):
            print(f"indices.index({i})=({indices.index(i)})")
            result += s[indices.index(i)]

        return result

    def decode(self, encoded: List[int], first: int) -> List[int]:
        """
        Explanation
        a XOR b = c, we know the values of a and c. we use the formula to find b -> a XOR c = b
        Complexity

        Time O(N)
        Space O(10)
        :param encoded:
        :param first:
        :return:
        """

        output = [first]

        for i in encoded:
            output.append(output[-1] ^ i)

        return output

    def xcanBeIncreasing(self, nums: List[int]) -> bool:
        flag = 1
        v = len(nums)
        while flag:
            count = 0
            for i in range(1, v):
                if nums[i - 1] < nums[i]:
                    pass
                else:
                    nums.remove(nums[i - 1])
                    count += 1
                    v -= 1
                    break

            if count > 0:
                flag = 1
            else:
                flag = 0

    def canBeIncreasing(self, nums: List[int]) -> bool:
        """
        Given an array of integers in the range between 1 and
        1000 inclusive, this program determines whether nums
        is strictly increasing or can be made that way with
        the removal of one element.

        The scenario of interest consists of an offending pair
        nums[k - 1] and nums[k] where nums[k] <= nums[k - 1],
        and nums[k - 2]. The solution is to remove one member
        of the offending pair. If nums[k] > nums[k - 2], we
        remove nums[k - 1]. Otherwise, we remove nums[k].
        For example:
            nums[k - 2]     nums[k - 1]     nums[k]
                5               6               4   remove nums[k]
                5               8               6   remove nums[k - 1]

        A special case occurs when the offending pair is the
        first two nums elements, nums[0] and nums[1]. in this
        case we simply remove the higher of the two values.
        The special case can be eliminated by padding nums with
        a zero at the start of the array.

        :param nums: array of integers, all between 1 and 1000
        :type nums: list[int]
        :return: True if nums is strictly increasing and can
                 be made that way with the removal of one element,
                 else False.
        :rtype: bool
        """

        """
        Initialize:
        - Pad nums to ease the handling of the edge case.
        - Store the length of nums (padded) in len_nums.
        - Initialize removed, a flag that indicates whether
          our allowed removal has been exercised.
        
        :param nums: 
        :return: 
        """

    def check_keywords(self, nums: List[int]) -> bool:
        """
                        for i in range(1, len(ns)):
                    if nums[i - 1] < nums[i]:
                        pass
                    else:
                        nums.remove(nums[i - 1])
                        v -= 1
                        count += 1
                        if count > 1:
                            return False
                        break
        :param nums:
        :return:
        """
        count = 0

        # Store the index of the element
        # that needs to be removed
        index = -1
        n = len(nums)
        # Traverse the range [1, N - 1]
        for i in range(1, len(nums)):

            # If arr[i-1] is greater than
            # or equal to arr[i]
            if (nums[i - 1] >= nums[i]):
                # Increment the count by 1
                count += 1

                # Update index
                index = i

        # If count is greater than one
        if (count > 1):
            return False

        # If no element is removed
        if (count == 0):
            return True

        # If only the last or the
        # first element is removed
        if (index == n - 1 or index == 1):
            return True

        # If a[index] is removed
        if (nums[index - 1] < nums[index + 1]):
            return True

        # If a[index - 1] is removed
        if (nums[index - 2] < nums[index]):
            return True

        return False

    def dsa_array(self):
        array1 = ar.array('i', [10, 20, 30, 40, 50])
        for x in array1:
            print(x)

        array1.insert(1, 60)
        array1.remove(40)
        # array1.index(40) print
        for x in array1:
            print(x)

        for x in array1:
            print(x)

    def dsa_dict(self):
        dicta = {'Name': 'Zara', 'Age': 7, 'Class': 'First'}
        del dicta['Name']  # remove entry with key 'Name'
        dicta.clear()  # remove all entries in dict
        del dicta

    def dsa_2D(self):
        z = [[11, 12, 5, 2], [15, 6, 10], [10, 8, 12, 5], [12, 15, 8, 6]]
        for r in z:
            for c in r:
                print(c, end=" ")
            print()
        # Updating Value
        T = [[11, 12, 5, 2], [15, 6, 10], [10, 8, 12, 5], [12, 15, 8, 6]]

        T[2] = [11, 9]
        del T[3]

        T[0][3] = 7
        for r in T:
            for c in r:
                print(c, end=" ")
            print()

        # DSA Matrix
        m = np.array([['Mon', 18, 20, 22, 17], ['Tue', 11, 18, 21, 18],
                      ['Wed', 15, 21, 20, 19], ['Thu', 11, 20, 22, 21],
                      ['Fri', 18, 17, 23, 22], ['Sat', 12, 22, 20, 18],
                      ['Sun', 13, 15, 19, 16]])
        m_r = np.append(m, [['Avg', 12, 15, 13, 11]], 0)

    def dsa_set(self):
        Days = set(["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"])
        Months = {"Jan", "Feb", "Mar"}
        Dates = {21, 22, 17}
        print(Days)
        print(Months)
        print(Dates)

        # Remove Item from set
        Days.discard("Sun")

        DaysA = set(["Mon", "Tue", "Wed"])
        DaysB = set(["Wed", "Thu", "Fri", "Sat", "Sun"])

        # Union
        AllDays = DaysA | DaysB
        print(AllDays)

        # Intersection
        AllDays = DaysA & DaysB
        print(AllDays)

        # Difference
        AllDays = DaysA - DaysB
        print(AllDays)

        # Compare Sets
        SubsetRes = DaysA <= DaysB
        SupersetRes = DaysB >= DaysA
        print(SubsetRes)
        print(SupersetRes)

    def dsa_chainmap(self):
        import collections

        dict1 = {'day1': 'Mon', 'day2': 'Tue'}
        dict2 = {'day3': 'Wed', 'day1': 'Thu'}

        res = collections.ChainMap(dict1, dict2)

        # Creating a single dictionary
        print(res.maps, '\n')

        print('Keys = {}'.format(list(res.keys())))
        print('Values = {}'.format(list(res.values())))
        print()

        # Print all the elements from the result
        print('elements:')
        for key, val in res.items():
            print('{} = {}'.format(key, val))
        print()

        # Find a specific value in the result
        print('day3 in res: {}'.format(('day1' in res)))
        print('day4 in res: {}'.format(('day4' in res)))

        # Map Reordering
        import collections

        dict1 = {'day1': 'Mon', 'day2': 'Tue'}
        dict2 = {'day3': 'Wed', 'day4': 'Thu'}

        res1 = collections.ChainMap(dict1, dict2)
        print(res1.maps, '\n')

        res2 = collections.ChainMap(dict2, dict1)
        print(res2.maps, '\n')


class Node:
    def __init__(self, dataval=None):
        self.dataval = dataval
        self.nextval = None


class Node2:
    def __init__(self, dataval):
        self.dataval = dataval
        self.next = None
        self.prev = None


class SLL:
    def __init__(self):
        self.headval = None

    def insert_between(self, insert_after, node_new):
        temp = insert_after.nextval
        new_node = Node(node_new)
        insert_after.nextval = new_node
        new_node.nextval = temp

    def print_list(self):
        printval = self.headval
        while printval is not None:
            print(printval.dataval)
            printval = printval.nextval


class SolutionList:
    def display(self, head):
        while head is not None:
            print(head.dataval, end='')
            head = head.nextval

    def remove(self, head, i_n):
        updated_head = head
        previous_node = None
        while updated_head.dataval is not i_n:
            previous_node = updated_head
            updated_head = updated_head.nextval
        if updated_head.nextval is None:
            previous_node.nextval = None
            return head
        if previous_node is None:
            updated_head = updated_head.nextval
            return updated_head
        previous_node.nextval = updated_head.nextval
        return head

    def insert_at_start(self, head, i_n):
        new_node = Node(i_n)
        new_node.nextval = head
        return new_node

    def insert(self, head, i_a, i_n):
        updated_head = head
        while updated_head.dataval is not i_a:
            updated_head = updated_head.nextval
        new_node = Node(i_n)
        temp = updated_head.nextval
        updated_head.nextval = new_node
        new_node.nextval = temp

        return head

    def add(self, head, data):
        updated_head = head
        if head is None:
            head = Node(data)
        else:
            while updated_head is not None:
                previous_node = updated_head
                updated_head = updated_head.nextval
            new_node = Node(data)
            previous_node.nextval = new_node

        return head


def dsa_linked_list():
    l1 = SLL()
    l1.headval = Node("Mon")
    l2 = Node("Tue")
    l3 = Node("Thur")
    l1.headval.nextval = l2
    l2.nextval = l3
    l1.print_list()
    l1.insert_between(l2, "Wednesday")
    l1.print_list()


def dsa_linked_past_four_years():
    mylist = SolutionList()
    T = int(input())
    head = None
    for i in range(T):
        data = int(input())
        head = mylist.add(head, data)
    mylist.display(head)
    # print("Provide number to be inserted")
    # inserted_number = int(input())
    # print("Provide number to be inserted after")
    # inserted_after = int(input())
    # head = mylist.insert(head, inserted_after, inserted_number)
    # mylist.display(head)
    # print("Provide number to be inserted at beginning")
    # inserted_number = int(input())
    # head = mylist.insert_at_start(head, inserted_number)
    # mylist.display(head)
    print("Provide number to be deleted")
    inserted_number = int(input())
    head = mylist.remove(head, inserted_number)
    mylist.display(head)


def digitSumInverse(sum, numberLength):
    count = 0
    for i in range(sum + 1):
        for j in range(sum + 1):
            if (i + j == sum and i >= sum):
                print(f'i:{i} j:{j}')


class Tracker:
    ls = {}
    i = 0

    def allocate(self, task):
        self.i += 1
        new = f'{task}{self.i}'
        if new in self.ls:
            self.ls[new] = int(self.ls.get(new)) + 1
        else:
            self.ls[new] = 1
        return f'{task}{self.ls.get(new)}'

    def deallocate(self, task):
        self.i += 1
        new = f'{task}{self.i}'
        if new in self.ls:
            self.ls.pop(new)

    def deallocate(self, task):
        self.i += 1
        new = f'{task}{self.i}'
        if new in self.ls:
            self.ls.pop(new)


def hostAllocation(queries):
    tracker = Tracker()
    ans = []
    for query in queries:
        task = query.split(' ')
        if task[0] == 'A':
            ans.append(tracker.allocate(task[1]))
        if task[0] == 'D':
            tracker.deallocate(task[1])
    return ans


class Stack:
    def __init__(self):
        self.stack = []

    def add(self, dataval):
        self.stack.append(dataval)

    def peek(self):
        if len(self.stack) >= 1:
            return self.stack[-1]
        return ValueError("No value to peek")

    def remove(self):
        if len(self.stack) <= 0:
            return "No element in the Stack"
        return self.stack.pop()


def dsa_stack(year):
    year = str(year)
    if int(year[len(year) - 1]) > 0:
        if len(year) < 4:
            return int(year[:1]) + 1
        else:
            return int(year[:2]) + 1
    else:
        if (len(year) < 4):
            return int(year[:1])
        else:
            return int(year[:2])


def solution2(inputString):
    print(inputString[::-1])
    return
    i = 0
    j = len(inputString) - 1
    if j == 0:
        return False
    while i < (len(inputString) / 2):
        if inputString[i] == inputString[j]:
            i += 1
            j -= 1
            continue
        else:
            return False
        i += 1
        j -= 1
    return True


def string_occurrence(s):
    count = dict()
    di = {}
    for i in s:
        if i in di:
            di[i] += 1
        else:
            di[i] = 1
    temp = sorted(di)
    s2 = sorted(temp, key=di.get, reverse=True)
    top_3 = s2[:3]
    for j in top_3:
        print(j, di[j])


def amazon(repository, customerQuery):
    check = customerQuery[:1] + ""
    ls = []
    for i in range(1, len(customerQuery)):
        check += customerQuery[i]
        all = []
        for repo in repository:
            if check == repo[:i + 1]:
                all.append(repo.lower())
        k = sorted(all)
        ls.append(k[:3])
    print(ls)
    return ls


def validMountainArray(arr: List[int]) -> bool:
    first_high = 0
    first_peak = 0
    no_high = 0
    i = 1

    for i in range(i, len(arr)):
        if (arr[i - 1] == arr[i]):
            return False
    i = 1
    for i in range(i, len(arr)):
        if (arr[i - 1] < arr[i]):
            first_high = 1
            break

    for i in range(i, len(arr)):
        if (arr[i - 1] > arr[i]):
            first_peak = 1
            break

    for i in range(i, len(arr)):
        if (arr[i - 1] < arr[i]):
            no_high = 1
            break

    if first_high == 1 and first_peak == 1 and no_high == 0:
        return True
    return False


def amazondone(codeList, shoppingCart):
    print("codeList: ", codeList)
    print("shoppingCart: ", shoppingCart)
    l = []
    for x in codeList:
        s = x.split(" ")
        for i in s:
            l.append(i)
    print("codeList: ", l)

    winner = 0
    ini = 0

    if len(l) != len(shoppingCart):
        return 0
    for ini in range(ini, len(l)):
        if shoppingCart[ini].lower() == l[0].lower():
            break

    for x in l:
        if (x == "anything"):
            ini += 1
            continue
        elif (x != shoppingCart[ini]):
            return 0
        ini += 1
    return 1


#
def sort_zeroes_and_ones(s):
    m = len(s) - 1
    for i in range(m, -1, -1):
        if s[i] != 0:
            s[m] = s[i]
            if (i != m):
                s[i] = 0
            m -= 1
    print(s)


def longestCommonPrefix(strs: List[str]) -> str:
    subs = []
    s = ""
    for i in strs:
        subs.append(len(i))
    n = strs[subs.index(min(subs))]

    for i in range(0, len(n)):
        s += n[i]
        for subs in strs:
            if s != subs[:i + 1]:
                return s
    print(n)
    return n


def bubble_sort_algo(s):
    for i in range(len(s) - 1):
        for j in range(len(s) - i - 1):
            if s[j] > s[j + 1]:
                temp = s[j + 1]
                s[j + 1] = s[j]
                s[j] = temp
    print(s)


def strStr(haystack: str, needle: str) -> int:
    j = 0
    if len(needle) == 0:
        return 0
    for i in range(len(needle), len(haystack) + 1):
        if needle == haystack[i - len(needle):i]:
            return i

    return -1


class Queue:
    def __init__(self):
        self.queue = []

    def add_queue(self, val):
        self.queue.append(val)

    def remove(self):
        if len(self.queue) > 1:
            return "Removed" + self.queue.pop(0)
        return "Queue is Empty. So cant remove"


def collectionsty():
    import collections
    dq = collections.deque(["Mon", "Tue", "Wed"])
    print(dq)
    dq.append("Thu")
    print(dq)
    dq.appendleft("Sun")
    print(dq)


def amazon_sde_2(buildingCount, routerLocation, routerRange):
    # [4, 6, 2, 5, 6, 2, 3],[2, 3, 7, 1, 3, 7, 4, 6, 1],[7, 0, 0, 2, 5, 2, 6, 1, 3]
    # [20] [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    #      [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0]
    # print(buildingCount, routerLocation, routerRange)
    # Write your code here
    c = min(buildingCount)
    if len(routerLocation) < c:
        return 0
    router = [0] * len(buildingCount)
    j = 0
    for m in routerLocation:
        for i in range(int(m) - routerRange[j], int(m) + routerRange[j] + 1):
            if (i >= 0 and i < len(buildingCount)):
                router[i] += 1
        j += 1
    g = 0
    for i in range(len(buildingCount)):
        if (router[i] < buildingCount[i]):
            g += 1
    return g


def permutation(str):
    findpermutation(str, "")


def findpermutation(str, prf):
    if len(str) == 0:
        print(str)
    else:
        for i in range(len(str)):
            rem = str[0:i] + str[i:]
            findpermutation(rem, prf + str[i])


def fibonaci(n):
    if n <= 0:
        return 0
    elif n == 1:
        return 1
    product = fibonaci(n - 1) + fibonaci(n - 2)
    return product


def ctci():
    """
    Perumation problem:
    Fibonacci problem:
    :return:
    """
    # permutation("string")
    print(fibonaci(8))


class BinaryNode:
    def __init__(self, node_value):
        self.left = None
        self.right = None
        self.node = node_value


class LeftViewTree:
    def get_result_now(self, root):
        if root is None:
            return
        print(root.node)
        self.get_result_now(root.left)
        self.get_result_now(root.right)

    def traverse_left(self, root):
        if root is None:
            return
        self.traverse_left(root.left)
        print(root.node)

    def traverse_right(self, root):
        if root is None:
            return

        print(root.node)
        self.traverse_right(root.right)


    def get_left_right_print(self, root):
        self.traverse_left(root)
        print("Traversed left")
        self.traverse_right(root.right)


if __name__ == "__main__":
    #             1
    #     2           3
    # 4       5    6       7
    obj = Solution()
    root = BinaryNode(1)
    root.left = BinaryNode(2)
    root.right = BinaryNode(3)
    root.left.left = BinaryNode(4)
    root.left.right = BinaryNode(5)
    root.right.left = BinaryNode(6)
    root.right.right = BinaryNode(7)
    root.right.right.right = BinaryNode(8)
    LeftViewTree().get_result_now(root)
    # LeftViewTree().get_left_right_print(root)
    # exit()
    # """
    # leetcode:
    # """
    # obj.shuffle_array([2, 5, 1, 3, 4, 7], 3)
    # obj.kidsWithCandies([2, 3, 5, 1, 3], 3)
    # obj.buildArray([5, 0, 1, 2, 3, 4])
    # obj.smallerNumbersThanCurrent([8, 1, 2, 2, 3])
    # obj.restoreString("codeleet", [4, 5, 6, 7, 0, 2, 1, 3])
    # obj.debug_restoreString("codeleet", [4, 5, 6, 7, 0, 2, 1, 3])
    # obj.decode([1, 2, 3], 1)
    # print(obj.canBeIncreasing([1, 2, 10, 5, 7]))
    # [105,924,32,968])
    # [1,2,10,5,7]
    # print(obj.check_keywords([2, 3, 1, 2]))
    # print(solution2("az"))
    print(string_occurrence("bbcccaaa"))
    exit()
    print(amazon(["mobile", "mouse", "moneypot", "monitor", "mousepad"], "mouse"))
    validMountainArray([2, 1, 2, 3, 5, 7, 9, 10, 12, 14, 15, 16, 18, 14, 13])
    print(amazondone(['kiwi', 'pear', 'jackfruit', 'orange', 'apple', 'banana', 'orange'] \
                     , ['kiwi', 'pear', 'jackfruit', 'orange', 'apple', 'mango', 'banana', 'orange']))
    sort_zeroes_and_ones([1, 0, 0])
    longestCommonPrefix(["a", "b"])
    bubble_sort_algo([1, 3, 4, 2, 5])
    strStr("abc", "c")

    """
    Start of Interview
    """
    digitSumInverse(5, 2)
    print(fibonaci(2))
    hostAllocation(["A u",
                    "A hk",
                    "A hk"])
    """
    Start of DSA
    """
    obj.dsa_array()
    obj.dsa_dict()
    obj.dsa_2D()
    obj.dsa_set()
    obj.dsa_chainmap()
    dsa_linked_list()
    dsa_linked_past_four_years()
    print(dsa_stack(45))
    AStack = Stack()
    AStack.add("Mon")
    AStack.add("Tue")
    AStack.peek()
    print(AStack.peek())
    AStack.add("Wed")
    AStack.add("Thu")
    print(AStack.peek())
    AStack.remove()
    AStack.remove()
    AStack.remove()
    AStack.remove()
    print(AStack.peek())
    q = Queue()
    q.add_queue("Mon")
    q.add_queue("Tue")
    q.add_queue("Wed")
    print(q.remove())
    print(q.remove())
    print(q.queue)
    collectionsty()
    amazon_sde_2([4, 6, 2, 5, 6, 2, 3], [2, 3, 7, 1, 3, 7, 4, 6, 1], [7, 0, 0, 2, 5, 2, 6, 1, 3])
    ctci()
