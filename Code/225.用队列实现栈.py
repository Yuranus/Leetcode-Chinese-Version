# -*- coding: utf-8 -*-
'''
使用队列实现栈的下列操作：

push(x) -- 元素 x 入栈
pop() -- 移除栈顶元素
top() -- 获取栈顶元素
empty() -- 返回栈是否为空
注意:

你只能使用队列的基本操作-- 也就是 push to back, peek/pop from front, size,
和 is empty 这些操作是合法的。
你所使用的语言也许不支持队列。
你可以使用 list 或者 deque（双端队列）来模拟一个队列 , 只要是标准的队列操作即可。
你可以假设所有操作都是有效的（例如, 对一个空的栈不会调用 pop 或者 top 操作）。
'''
class Queue:
    def __init__(self):
        self.queue=[]

    def size(self):
        '''return the size of the queue'''
        return len(self.queue)

    def isempty(self):
        '''return the bool value indicating whether the queue is empty'''
        return self.size()==0

    def enqueue(self,val):
        '''Enqueue the value val into the queue from rear'''
        self.queue.append(val)

    def dequeue(self):
        '''return the dequeued value from the head of queue'''
        return self.queue.pop(0)

    def peek(self):
        '''return the value of the head of queue'''
        return self.queue[0]

class MyStack:
    def __init__(self):
        """
        Initialize your data structure here.
        """
        self.stack=Queue()

    def push(self, x):
        """
        Push element x onto stack.
        :type x: int
        :rtype: void
        """
        self.stack.enqueue(x)

    def pop(self):
        """
        Removes the element on top of the stack and returns that element.
        :rtype: int
        """
        for i in range(self.stack.size()-1):
            self.stack.enqueue(self.stack.dequeue())

        return self.stack.dequeue()

    def top(self):
        """
        Get the top element.
        :rtype: int
        """
        for i in range(self.stack.size()-1):
            self.stack.enqueue(self.stack.dequeue())

        value=self.stack.peek()
        self.stack.enqueue(self.stack.dequeue())
        return value


    def empty(self):
        """
        Returns whether the stack is empty.
        :rtype: bool
        """
        return self.stack.isempty()

# Your MyStack object will be instantiated and called as such:
obj = MyStack()
obj.push(3)
obj.push(5)
obj.push(6)
param_2 = obj.pop()
print(param_2)
param_3 = obj.top()
print(param_3)
param_4 = obj.empty()
print(param_4)
#思路：用队列实现栈的方法top和pop必须遍历队列到末尾，方法就是先出队再进队，最后一个就是栈顶元素#
#效率：42.90%#