from yahoo import search
from bs4 import BeautifulSoup

for url in search("what does the fox say?"):
    print(url)