# -*- coding: utf-8 -*-
'''
给定一个大小为 n 的数组，找出其中所有出现超过 ⌊ n/3 ⌋ 次的元素。

说明: 要求算法的时间复杂度为 O(n)，空间复杂度为 O(1)。

示例 1:

输入: [3,2,3]
输出: [3]
示例 2:

输入: [1,1,1,3,3,2,2,2]
输出: [1,2]
'''
class Solution:
    def majorityElement(self, nums):
        """
        :type nums: List[int]
        :rtype: List[int]
        """
        if len(nums)==0:
            return []

        length=len(nums)

        num1,num2,time1,time2=None,None,0,0
        for num in nums:
            if num1==num:
                time1+=1
            elif num2==num:
                time2+=1
            elif time1==0:
                num1,time1=num,1
            elif time2==0:
                num2,time2=num,1
            else:
                time1,time2=time1-1,time2-1

        return [n for n in (num1,num2) if n is not None and nums.count(n)>length/3]

s=Solution()
print(s.majorityElement([3,2,3]))
print(s.majorityElement(([1,1,1,3,3,2,2,2])))
#思路：在一个数组中出现超过三分之一次的元素，这样的元素最多只能有两个，超过两个就与命题相矛盾。#
# 摩尔投票法：在每一轮投票过程中，从数组中找出一对不同的元素，将其从数组中删除。
# 这样不断的删除直到无法再进行投票，如果数组为空，则没有任何元素出现的次数超过该数组长度的一半。
# 如果只存在一种元素，那么这个元素则可能为目标元素。#
#效率：86.36%#
