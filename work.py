"""import numpy as np
import pandas as pd
import jieba
from imageio import imread
import matplotlib.pyplot as plt
from PIL import Image
from wordcloud import WordCloud, STOPWORDS
"""
def q1():
    x = np.linspace(0, 2, 100)

    plt.plot(x,
             (1/np.exp(x))*np.sin(x),
             label='(1/np.exp(x))*np.sin(x)')
    plt.plot(x,
             (1/np.exp(x))*np.cos(x),
             'orange',
             label='(1/np.exp(x))*np.cos(x)')
    plt.ylim(-0.2, 1.15)
    plt.legend()

    plt.show()

def q2():
    labels = ['USA', 'China', 'European Union', 'Saudi Arabia', 'Russia', 'India',
              'France', 'United Kindom', 'Japan', 'Other countries']
    sizes = [610, 235, 200, 69.5, 66.3, 63.9, 57.8, 47.2, 45.4, 281.9]
    explode = (0, 0.25, 0, 0, 0, 0, 0, 0, 0, 0)

    plt.pie(sizes,
            explode=explode,
            labels=labels,
            autopct='%1.1f%%')

    plt.legend()

    plt.show()

def q3():
    values = [1093997.267, 1211346.87, 1339395.719, 1470550.015, 1660287.966, 1955347.005,
        2285965.892, 2752131.773,3552182.312, 4598206.091, 5109953.609, 6100620.489,
        7572553.837, 8560547.315, 9607224.482, 10482372.11,11064666.28, 11190992.55, 12237700.48]
    x = [1999, 2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007,
         2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017]

    total_width, n = 0.8, 1
    width = total_width / n

    plt.bar(x,
            values,
            width=width,
            label="China's GDP",
            color=['black', 'red', 'green', 'blue', 'cyan'])

    plt.grid()
    plt.legend()

    plt.show()

def q4():
    treatment = pd.read_csv('treatment1.csv', index_col=0)

    sns.set_style('darkgrid')
    sns.stripplot(data=treatment, x='treatment', y='fluorescence', jitter=True, alpha=0.5)
    sns.boxplot(data=treatment, x='treatment', y='fluorescence')

def q5():
    cappuccino = np.array(Image.open("cappuccino.png"))
    txt = open('17 cards.txt', 'r', encoding='utf-8').read()
    txtout = "/".join(jieba.cut(txt))
    font = 'C:\\Windows\\Fonts\\simfang.ttf'
    stopwords = set(STOPWORDS)

    wc = WordCloud(font_path=font,
                   background_color="white",
                   width=800,
                   height=600,
                   max_words=200,
                   max_font_size=80,
                   stopwords=stopwords,
                   mask=cappuccino)
    wc.generate(txtout)

    wc.to_file('wc.png')

if __name__ == '__main__':
    pass