'''
给定两个字符串形式的非负整数 num1 和num2 ，计算它们的和。

注意：

num1 和num2 的长度都小于 5100.
num1 和num2 都只包含数字 0-9.
num1 和num2 都不包含任何前导零。
你不能使用任何內建 BigInteger 库， 也不能直接将输入的字符串转换为整数形式。
'''
class Solution:
	def add(self,num1,num2):
		'''num1 is longer, num2 is shorter'''
		flag=False
		string=[]
		temp=[]
		if len(num1)>len(num2):
			temp=num1[:len(num1)-len(num2)]
		num1=num1[len(num1)-len(num2):]

		for i in reversed(range(len(num2))):
			if not flag:
				res=int(num1[i])+int(num2[i])
				if res>=10:
					string.insert(0,str(res-10))
					flag=True
				else:
					string.insert(0,str(res))
					flag=False
			else:
				res=int(num1[i])+int(num2[i])+1
				if res>=10:
					string.insert(0,str(res-10))
					flag=True
				else:
					string.insert(0,str(res))
					flag=False

		if len(temp)!=0:
			for i in reversed(range(len(temp))):
				if not flag:
					string.insert(0,temp[i])
					flag=False
				else:
					res=int(temp[i])+1
					if res>=10:
						string.insert(0,str(res-10))
						flag=True
					else:
						string.insert(0,str(res))
						flag=False

		if flag:
			string.insert(0,'1')

		return ''.join(string)

	def addStrings(self, num1, num2):
		"""
		:type num1: str
		:type num2: str
		:rtype: str
		"""
		if len(num1)==0 and len(num2)==0:
			return ''
		if len(num1)==0:
			return num2
		if len(num2)==0:
			return num1

		if len(num1)>=len(num2):
			return self.add(num1,num2)
		else:
			return self.add(num2,num1)

s=Solution()
print(s.addStrings('123456','57483839'))
#思路：和二进制相加一个方法，看进位即可#
#效率：47.71%#

