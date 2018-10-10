# -*- coding: utf-8 -*-
'''
给定一个无重复元素的有序整数数组，返回数组区间范围的汇总。

示例 1:

输入: [0,1,2,4,5,7]
输出: ["0->2","4->5","7"]
解释: 0,1,2 可组成一个连续的区间; 4,5 可组成一个连续的区间。
示例 2:

输入: [0,2,3,4,6,8,9]
输出: ["0","2->4","6","8->9"]
解释: 2,3,4 可组成一个连续的区间; 8,9 可组成一个连续的区间。
'''
class Solution:
    def summaryRanges(self, nums):
        """
        :type nums: List[int]
        :rtype: List[str]
        """
        if len(nums)==0:
            return []

        if len(nums)==1:
            return [str(nums[0])]

        result=[]
        start=0
        end=0
        for i in range(1,len(nums)):
            if nums[i]==nums[i-1]+1:
                end+=1
            else:
                if end!=start:
                    result.append(str(nums[start])+'->'+str(nums[end]))
                else:
                    result.append(str(nums[start]))
                start=i
                end=i

        if end != start:
            result.append(str(nums[start]) + '->' + str(nums[end]))
        else:
            result.append(str(nums[start]))

        return result

s=Solution()
print(s.summaryRanges([0,1,2,4,5,7]))
print(s.summaryRanges([0,2,3,4,6,8,9]))
#思路：思路简单，直接设置两个标识位，start指示连续递增数字开始的位置，end代表结束#
#注意结束循环也要判断一次start和end#
#效率：54.24%#


