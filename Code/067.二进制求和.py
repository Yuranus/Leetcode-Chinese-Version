# -*- coding: utf-8 -*-
'''
给定两个二进制字符串，返回他们的和（用二进制表示）。

输入为非空字符串且只包含数字 1 和 0。

示例 1:

输入: a = "11", b = "1"
输出: "100"
示例 2:

输入: a = "1010", b = "1011"
输出: "10101"
'''
class Solution:
    def add(self,a,b):
        '''a is longer, b is shorter'''
        flag=False#flag为进位标识符，True为有进位，False为无进位
        #分几种情况：#
        #1：前面没有进位，现在有进位；#
        #2：前面有进位，现在没有进位；#
        #3：前面没有进位，现在也没有进位；#
        #4：前面有进位，现在也有进位。#
        string=[]
        a_temp=[]
        if len(a)-len(b)!=0:#必须要对数组拆分
            a_temp=a[:len(a)-len(b)]
            a=a[len(a)-len(b):]

        for i in reversed(range(len(b))):
            if not flag:#前面没有进位的情况
                add=int(a[i])+int(b[i])
                if add==2:#情况1
                    string.insert(0,'0')
                    flag=True
                elif add==1:#情况3
                    string.insert(0,'1')
                    flag=False
                elif add==0:#情况3
                    string.insert(0,'0')
                    flag=False
            else:#前面有进位的情况
                add=int(a[i])+int(b[i])+1
                if add==3:#情况4
                    string.insert(0,'1')
                    flag=True
                elif add==2:#情况4
                    string.insert(0,'0')
                    flag=True
                elif add==1:#情况2
                    string.insert(0,'1')
                    flag=False

        if len(a_temp)!=0:#对于剩余的a中的数字
            a=a_temp
            for i in reversed(range(len(a))):
                if not flag:
                    string=[x for x in a[:i+1]]+string
                    break
                else:
                    add=int(a[i])+1
                    if add==2:
                        string.insert(0,'0')
                        flag=True
                    elif add==1:
                        string.insert(0,'1')
                        flag=False

        #对于最后一位有进位#
        if flag:
            string.insert(0,'1')

        return ''.join(string)

    def addBinary(self, a, b):
        """
        :type a: str
        :type b: str
        :rtype: str
        """
        if len(a)==0 and len(b)==0:
            return ''
        if len(a)==0:
            return b
        if len(b)==0:
            return a

        if len(a)<len(b):
            return self.add(b,a)
        else:
            return self.add(a,b)

s=Solution()
print(s.addBinary('1010','1011'))
print(s.addBinary('0','0'))
print(s.addBinary('100','110010'))
print(s.addBinary('101111','10'))
#思路：本题思路和66题一样的，只不过是b长度不是1了，所以就判断a和b长度的关系，同时判断进位，有4种情况#
#注意当最后的时候，a还有剩余的话要继续判断，同时完全结束之后也要进行判断之前有没有进位#
#效率：27.94%#




