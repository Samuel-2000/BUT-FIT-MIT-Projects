import requests
import sys
from bs4 import BeautifulSoup

starting_url = "https://www.altronics.com.au/audio-visual/"

item_urls = []

current_page = starting_url

while True:
    response = requests.get(current_page)

    if response.status_code == 200:
        soup = BeautifulSoup(response.content, 'html.parser')

        for item in soup.find_all('div', class_='grid-item'):
            item_url = item.find('a', class_='')['href']
            item_urls.append(item_url)
    
    if(len(item_urls) > 200):
        break
   
    next_page_link = soup.find_all('a', class_='pull-right')[1]
    if next_page_link:
        current_page = starting_url +  next_page_link['href']
    else:
        break


for url in item_urls:
    sys.stdout.write(f"{url}\n")
        