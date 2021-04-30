import requests
from bs4 import BeautifulSoup

def spider():
    url = 'https://html.com/tags/table/'
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 '
                    '(KHTML, like Gecko) Chrome/88.0.4324.182 Safari/537.36'
    }
    response = requests.post(url, headers=headers)

    bsobj = BeautifulSoup(response.text, 'lxml')
    tables = bsobj.find('table')
    text = []

    for table in tables:
        for tr in table.find_all('tr'):
            for td in tr.find_all('td'):
                text.append(td.get_text())

    print(text)

spider()