# -*- coding: utf-8 -*-
'''
给定一个整型数组，在数组中找出由三个数组成的最大乘积，并输出这个乘积。

示例 1:

输入: [1,2,3]
输出: 6
示例 2:

输入: [1,2,3,4]
输出: 24
'''
class Solution:
    def maximumProduct(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        if len(nums)<3:
            return 0

        if len(nums)==3:
            return nums[0]*nums[1]*nums[2]
        nums.sort()
        if nums[-1]<0:#全负
            return nums[0]*nums[1]*nums[2]
        if nums[0]>=0:#全正
            return nums[-1]*nums[-2]*nums[-3]
        else:#有正有负
            result1=nums[0]*nums[1]*nums[-1]
            result2=nums[-1]*nums[-2]*nums[-3]
            return max(result1,result2)

s=Solution()
print(s.maximumProduct([1,2,3]))
print(s.maximumProduct([1,2,3,4]))
print(s.maximumProduct([-4,-3,-2,-1,60]))
#思路：先排序，从小到大，再分情况讨论：全负，最大就在前三个；全正，最大就在后面三个，有正有负，#
#最大要么在前面两个负的和最后一个，要么在后面三个#
#效率：83.24%#