# -*- coding: utf-8 -*-
'''
给定两个字符串, A 和 B。

A 的旋转操作就是将 A 最左边的字符移动到最右边。 例如, 若 A = 'abcde'，在移动一次之后结果就是'bcdea' 。
如果在若干次旋转操作之后，A 能变成B，那么返回True。

示例 1:
输入: A = 'abcde', B = 'cdeab'
输出: true

示例 2:
输入: A = 'abcde', B = 'abced'
输出: false
'''
class Solution:
    def rotateString(self, A, B):
        """
        :type A: str
        :type B: str
        :rtype: bool
        """
        if len(A)!=len(B):
            return False
        if len(A)==0:
            return True
        if len(A)==1:
            if A[0]==B[0]:
                return True
            return False
        if A==B:
            return True

        for i in range(len(A)-1):
            string=A[i+1:]+A[:i+1]
            if string==B:
                return True

        return False

s=Solution()
print(s.rotateString('abcde','cdeab'))
print(s.rotateString('abcde','abced'))
#思路：操作全部结束的标志是，A移回本身，每次得到新的字符串，重新判断即可#
#效率：88.66%#