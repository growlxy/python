import time
import random
import requests
from bs4 import BeautifulSoup

class Suki_chara:
    def __init__(self):
        self.number = 0
        self.t = time.perf_counter()

    def __new__(cls, *args, **kwargs):
        print('大好き！')
        return super().__new__(cls)

    def ticket(self):
        sleep_time = round(random.random(), 2) + 2
        time.sleep(sleep_time)

        url = 'http://www.hook-net.jp/ikoi/character/counterC.php'

        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 '
                          '(KHTML, like Gecko) Chrome/90.0.4430.93 Safari/537.36'
        }
        response = requests.get(url, headers=headers)

        bsobj = BeautifulSoup(response.text, 'lxml')
        number = bsobj.find('p').get_text()

        self.number = number

    @property
    def timer(self):
        now_t = time.perf_counter()
        return round(now_t - self.t, 2)

if __name__ == '__main__':
    iku = Suki_chara()
    while True:
        iku.ticket()
        print(f'suki + 1, total: {iku.number}'
              f' 郁ちゃん！大好きです！'
              f' total running time: {iku.timer}s')