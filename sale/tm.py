li=[[1,2],[3,4]]
for i in map(list, zip(*li)):
	print(i)