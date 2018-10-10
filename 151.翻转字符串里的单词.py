# -*- coding: utf-8 -*-
'''
给定一个字符串，逐个翻转字符串中的每个单词。

示例:

输入: "the sky is blue",
输出: "blue is sky the".
说明:

无空格字符构成一个单词。
输入字符串可以在前面或者后面包含多余的空格，但是反转后的字符不能包括。
如果两个单词间有多余的空格，将反转后单词间的空格减少到只含一个。
'''
class Solution(object):
	def reverseWords(self, s):
		"""
        :type s: str
        :rtype: str
        """
		if len(s)==0 or len(s)==1:
			return s

		if ' '*len(s)==s:
			return s[:0]

		s=s.strip().split()

		s=s[::-1]

		return ' '.join(s)
		'''
		length=len(s)
		mid=length//2
		for i in range(mid):
			s[i],s[length-1-i]=s[length-1-i]+' ',s[i]+' '

		if length%2!=0:
			s[mid]+=' '

		s=''.join(s)
		return s[:-1]'''

s=Solution()
print(s.reverseWords('the sky is blue'))
print(s.reverseWords('I know you love me'))
print(s.reverseWords('    '))
#Python版本没过，不知道是什么原因，一个测试用例就是全是空格的过不去，采用了网上的C#版本，效率：100%#
#代码如下：#
'''
public class Solution
{
    public string ReverseWords(string s)
    {
        StringBuilder sb = new StringBuilder();
        s = s.Trim();
        var words = s.Split(new char[] {' ','\t'}, StringSplitOptions.RemoveEmptyEntries);
        for (int i = words.Length; i > 0; i--)
        {
            foreach (var ch in words[i-1])
            {
                sb.Append(ch);
            }
            sb.Append(" ");
        }
        return sb.ToString().Trim();
    }
}

'''




