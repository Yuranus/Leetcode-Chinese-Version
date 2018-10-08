# -*- coding: utf-8 -*-
'''
给定一个整数类型的数组 nums，请编写一个能够返回数组“中心索引”的方法。

我们是这样定义数组中心索引的：数组中心索引的左侧所有元素相加的和等于右侧所有元素相加的和。

如果数组不存在中心索引，那么我们应该返回 -1。如果数组有多个中心索引，那么我们应该返回最靠近左边的那一个。

示例 1:

输入:
nums = [1, 7, 3, 6, 5, 6]
输出: 3
解释:
索引3 (nums[3] = 6) 的左侧数之和(1 + 7 + 3 = 11)，与右侧数之和(5 + 6 = 11)相等。
同时, 3 也是第一个符合要求的中心索引。
示例 2:

输入:
nums = [1, 2, 3]
输出: -1
解释:
数组中不存在满足此条件的中心索引。
'''
class Solution:
    def pivotIndex(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        if len(nums)==0:
            return -1
        if len(nums)==1:
            return 0

        add_all=sum(nums)
        add_left=0
        for i in range(len(nums)):
            add_all-=nums[i]
            if add_left==add_all:
                return i
            add_left+=nums[i]

        return -1

s=Solution()
print(s.pivotIndex([1, 7, 3, 6, 5, 6]))
print(s.pivotIndex([1,2,3]))
#思路：不能二分法，因为无序，也不能设置左右指针，因为数组正负不确定#
#使用最简单的：先求数组之和，遍历一次就减去当前值，判断左右两边是否相等，注意下标变化。#
#效率：81.69%#

