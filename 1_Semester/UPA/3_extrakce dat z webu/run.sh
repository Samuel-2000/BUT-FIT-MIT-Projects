#! /bin/bash

python3 items_urls.py > url_test.txt
python3 item.py -f url_test.txt -c 10