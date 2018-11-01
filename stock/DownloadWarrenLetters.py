import sys
import urllib.request
import os


webaddress = "http://www.berkshirehathaway.com//letters"
despath_value= "d:\\warrenbuffett";

def getdata(webaddr, despath, desname):
	req = urllib.request.Request(webaddr+"//"+desname)
	res =  urllib.request.urlopen( req )
	##headers = str(res.info())
	##print(headers)
	html=res.read()
	filename = despath + "\\" + desname
	print("write file:" + filename)
	writeFile = open(filename,'w+b')
	writeFile.write(html)
	writeFile.close()
	print("finished")

if not os.path.isdir(despath_value):
	os.makedirs(despath_value)
for year in range(1977, 2018):
	if year <= 1997:
		tmp = str(year) + ".html"
	elif year <= 1998:
		tmp = str(year) + "pdf.pdf"
	elif year <= 1999:
		tmp = "final" + str(year) + "pdf.pdf"
	elif year <= 2002:
		tmp = str(year) + "pdf.pdf"
	else:
		tmp = str(year) + "ltr.pdf"
	getdata(webaddress, despath_value, tmp)