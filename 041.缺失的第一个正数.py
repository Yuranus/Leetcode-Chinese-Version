# -*- coding: utf-8 -*-
'''
给定一个未排序的整数数组，找出其中没有出现的最小的正整数。

示例 1:

输入: [1,2,0]
输出: 3
示例 2:

输入: [3,4,-1,1]
输出: 2
示例 3:

输入: [7,8,9,11,12]
输出: 1
说明:

你的算法的时间复杂度应为O(n)，并且只能使用常数级别的空间。
'''
class Solution:
    def firstMissingPositive(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        if len(nums)==0:
            return 1

        left=0
        right=len(nums)
        while left<right:
            if nums[left]==left+1:
                left+=1
            elif nums[left]<left+1 or nums[left]>right or nums[nums[left]-1]==nums[left]:
                right-=1
                nums[left]=nums[right]
            else:
                nums[nums[left]-1],nums[left]=nums[left],nums[nums[left]-1]

        return left+1

s=Solution()
print(s.firstMissingPositive([1,2,0]))
print(s.firstMissingPositive([3,4,-1,1]))
print(s.firstMissingPositive([7,8,9,11,12]))
print(s.firstMissingPositive([1]))
#效率：41.54%#
#思路：更换每个数字的位置到其指定的下标上，即，下标位置+1=元素，但是更换之后有可能有些位置#
#会乱，所以简单的方法就是设置前后指针，交换之后左边不变，右边遍历，直到两个指针相遇，则left+1就是#
#那个元素#


