#!/usr/bin/python3
import pymysql

class DB:
    def __init__(self, data):
        self.data = data

    def work(self):
        db = pymysql.connect(host='127.0.0.1', user='root', password="", database ='master_system')
        return db

    def select(self, data):
        db = self.work()
        cursor = db.cursor()
        cursor.execute(self.data)
        data = cursor.fetchone()
        print(f'{data}')
        self.end()

    def insert(self, data):
        db = self.work()
        cursor = db.cursor()
        cursor.execute(self.data)
        db.commit()
        self.end()

    def delete(self, data):
        db = self.work()
        cursor = db.cursor()
        cursor.execute(self.data)
        db.commit()
        self.end()

    def end(self):
        cursor.close()
        db.close()

    def __del__(self):
        print('执行完毕')


if __name__ == '__main__':
    sql = """insert into student values('1', 'wang', '男', '15', '1')"""
    db = DB(sql)
    db.insert()