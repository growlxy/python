import requests
from bs4 import BeautifulSoup

def spider():
    for page in range(8560, 8565):
        for n in range(2):
            next = '_1'
            if n  == 0:
                url = 'https://www.book900.com/4_4946/' + str(page)
            else:
                url = 'https://www.book900.com/4_4946/' + str(page) + next
            response = requests.get(url)

            bsobj = BeautifulSoup(response.text, 'lxml')
            title = str(bsobj.find('h1'))
            data = str(bsobj.find(id='content'))

            retitle = ['<h1>', '(2/2)</h1>']
            for i in retitle:
                title = title.replace(i, '')


            redata = ['\n', '\t', '\xa0', '<br/>', '<div id="content">', '<br-->', '&gt;', '</div>', '--', ' </br>',
                      '<p class="to_nextpage"><a href="/4_4946/' + str(page) + next + '" rel="next">'
                      '本章未完，点击下一页继续阅读</a></p>']
            for i in redata:
                data = data.replace(i,'')

            if n == 0:
                text_1 = data
            else:
                text_2 = data.lstrip('\r')
                text_2 = text_2.rstrip('。')

        fp = open(f'note\\{title}.txt', 'w', encoding='utf-8')
        fp.write(title)
        fp.write('\n')
        fp.write(text_1)
        fp.write('\n')
        fp.write(text_2)
        fp.write('\n')
        print(title + ' 下载完成！')
        fp.close()

spider()