def two_sum_1():
    """
    Optimized
    :return:
    """
    nums = [7, 2, 13, 11]
    target = 9
    n = len(nums)
    check = {}
    for i in range(n):
        if target - nums[i] in check:
            print([check[target - nums[i], i]])
            return
        check[nums[i]] = i

    """
        Unoptimized
        :return:
    """
    # nums = [7, 2, 13, 11]
    # target = 24
    for i in range(n - 1):
        for j in range(i + 1, n):
            if nums[i] + nums[j] == target:
                print([i, j])
                return


def median_of_two_sorted_arrays_2():
    """
    Unoptimized
    :return:
    """
    arr1 = [2, 3, 5, 8]
    arr2 = [10, 12, 14, 16, 18, 20]
    n = len(arr1)
    m = len(arr2)
    c1 = 0
    c2 = 0
    i = 0
    j = 0
    for count in range(int((n + m) / 2) + 1):
        c2 = c1
        if i < n and j < m:
            if arr1[i] > arr2[j]:
                c1 = arr2[j]
                j += 1
            else:
                c1 = arr1[i]
                i += 1
        elif i < n:
            c1 = arr1[j]
            i += 1
        else:
            c1 = arr2[j]
            j += 1
    if ((n + m) % 2 == 0):
        print("C1:", c1)
        print("C2:", c2)
        print((c1 + c2) / 2)
    else:
        print(c1)
    """
    Intuitive
    :return: 
    """
    # arr1=[-5, 3, 6, 12, 15]
    # arr2=[-12, -10, -6, -3, 4, 10]
    l = arr1 + arr2
    l.sort()
    i = len(l)
    if len(l) % 2 != 0:
        return l[int(i / 2)]
    else:
        return ((l[(int(i / 2)) - 1]) + (l[int(i / 2)])) / 2


def hyphen_to_underscore(m):
    print(m.replace("-", "_"))
    x=123
    if (x < 0):
        x = -x
    x = str(x)
    x = int(x[::-1])
    if (x > 2147483647):
        return 0
    if (x < 0):
        return x * -1
    return x


if __name__ == '__main__':
    # to_do merge_sort()
    hyphen_to_underscore("median-of-two-sorted-arrays-2")
    # two_sum_1()
    median_of_two_sorted_arrays_2()
    # ()
