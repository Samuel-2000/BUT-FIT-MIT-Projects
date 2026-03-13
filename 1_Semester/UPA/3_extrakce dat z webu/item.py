import time
import regex as re
import sys
import argparse
from bs4 import BeautifulSoup
from urllib.request import urlopen

parser = argparse.ArgumentParser()
parser.add_argument('-f', '--file', type=str, default="urls.txt", help='Name of a file with urls')
parser.add_argument('-c', '--count', type=int, help='How many items should be scrapped')

args = parser.parse_args()

filename = args.file
count = args.count


with open(filename, 'r') as item_urls:
    
    if count:
        item_urls = item_urls.readlines()[:count]
        
    for item_url in item_urls:
        
        response = urlopen(item_url)

        html = response.read().decode("utf-8")
        soup = BeautifulSoup(html, 'html.parser')
        name = soup.find('h1').text.strip()
        

        price_text=""
        price = soup.find('span', class_='top-price')
        if price:
            price_text= price.text.strip()
        
        spec = soup.find('ul', class_='productspecs')
        barcode_text= ""
        weight_text=""
        size_text=""

        if spec:
            barcode = spec.find(string=re.compile("Barcode:"))
            if barcode:
                barcode_text=barcode.replace("Barcode: ", "").strip()
                
            weight = spec.find(string=re.compile("Shipping Weight:"))
            if weight:
                weight_text=weight.replace("Shipping Weight: ", "").strip()
                
            size = spec.find(string=re.compile("CARTON:"))
            if size:
                size_text=size.replace("CARTON: ", "").strip()

        
        sys.stdout.write(f"{item_url.strip()}\t{name}\t{price_text}\t{barcode_text}\t{weight_text}\t{size_text}\n")
        time.sleep(0.5)
