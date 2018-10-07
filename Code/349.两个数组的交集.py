# -*- coding: utf-8 -*-
'''
给定两个数组，编写一个函数来计算它们的交集。

示例 1:

输入: nums1 = [1,2,2,1], nums2 = [2,2]
输出: [2]
示例 2:

输入: nums1 = [4,9,5], nums2 = [9,4,9,8,4]
输出: [9,4]
说明:

输出结果中的每个元素一定是唯一的。
我们可以不考虑输出结果的顺序。
'''
class Solution:
    def final(self,nums1,nums2):
        '''nums1 is longer, nums2 is shorter'''
        result=[]
        for num in nums2:
            if num in nums1:
                result.append(num)

        return result

    def intersection(self, nums1, nums2):
        """
        :type nums1: List[int]
        :type nums2: List[int]
        :rtype: List[int]
        """
        if len(nums1)==0 or len(nums2)==0:
            return []

        #Method one : use Python built-in function#
        #return list(set(nums1).intersection(nums2))
        #Method two : create new method#
        nums1=set(nums1)
        nums2=set(nums2)

        if len(nums1)>=len(nums2):
            return self.final(nums1,nums2)
        else:
            return self.final(nums2,nums1)


s=Solution()
print(s.intersection([4,9,5],[9,4,9,8,4]))
#思路：可以使用python内置函数，也可以将短的数组中的数字遍历，查看长的数组中有无次数即可#
#效率：49.49%#
