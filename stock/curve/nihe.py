import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time
import math
a=pd.read_csv('GOOGL.csv')

def gettime(tss1):
	return int(time.mktime(time.strptime(tss1, "%Y-%m-%d")))

def rotate(angle,valuex,valuey):
    rotatex = math.cos(angle)*valuex -math.sin(angle)*valuey
    rotatey = math.cos(angle)*valuey + math.sin(angle)* valuex
    rotatex = rotatex.tolist()
    rotatey = rotatey.tolist()
    xy = rotatex + rotatey
    return xy




# x = a['Date'].apply(gettime).values
x=np.linspace(0, 1000, len(a['Adj Close'].tolist()))
y = a['Adj Close'].values
x1=x
a1 = rotate(math.pi/4,x,y)



z1 = np.polyfit(x, y, 4)#用3次多项式拟合
p1 = np.poly1d(z1)
print(p1) #在屏幕上打印拟合多项式
# x=np.linspace(0, 1)
# x=np.linspace(0, 200, 200)
yvals=p1(x)
yvals=p1(x1)
d1=p1.deriv() 
yd1=d1(x)
# bb=pd.DataFrame({'x':x,'y':y,'yvals':yvals,'yd1':yd1})
# bb.to_csv('sum.csv')





li=[]
for i in range(len(yvals)):
    if i != (len(yvals)-1):
        tmp=(yvals[i+1]-yvals[i])/(1+yvals[i+1]*yvals[i])
        # differ=np.arctan(tmp)
        differ=tmp
        li.append(differ)
li.append(0)        
d1=p1.deriv() 
d2=p1.deriv().deriv()  
print(d1)
print(d2)
yd1=d1(x)
yd2=d2(x)
k=np.diff(yd1) 
print(yd1)
print(k)
k=np.append(k,0)
def getk(yd1,yd2):
    li_tmp=[]
    for i in range(len(yd1)):
        
        li_tmp.append(np.abs(yd2[i])/math.pow((1+math.pow(yd1[i],2)),1.5))
    return li_tmp

def abss(d1):
	return np.power((1+np.power(d1,2)),1.5)
bb=np.abs(d1)


print(bb)
print(abss)

# aa=bb/cc

k = getk(yd1,yd2)

#也可以使用yvals=np.polyval(z1,x)
plot1=plt.plot(x1, y, '*',label='original values')
plot2=plt.plot(x1, yvals, '*',label='polyfit values')
# plot2=plt.plot(x, yd2, '*',label='polyfit values')
plt.show()

# # plot2=plt.plot(x, k, '*',label='polyfit values')
# plt.xlabel('x axis')
# plt.ylabel('y axis')
# plt.legend(loc=4)#指定legend的位置,读者可以自己help它的用法
# plt.title('polyfitting')
# plt.show()
# # plt.savefig('p1.png')

