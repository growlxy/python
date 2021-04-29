import random

class Dice:
    def __init__(self, times):
        self.times = times
    # 骰子投的次数
    def roll(self):
        return random.randint(1, 6)
    # 投一次骰子
    def probability(self):
        L = [0, 0, 0, 0, 0, 0]
        for i in range(1, self.times + 1):
            n = self.roll()
            L[n - 1] += 1
    # 将数字输入列表
        for i in range(0, 6):
            L[i] = str(L[i] / self.times * 100) + '%'
    # 将统计出的数字转换成百分数
        return L

    @property
    def result(self):
        num = self.probability()
        for i in range(1, 7):
            print(str(i) + ': ' + num[i - 1])
    # 输入结果

if __name__ == '__main__':
    dice = Dice(100)
    dice.result
