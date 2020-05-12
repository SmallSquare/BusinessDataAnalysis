# coding=utf-8

# Code by SmallSquare, 2020/5.
# Provide an easy access to Mysql.
import chardet
import pymysql
import private_settings

"""
"private_settings.py" is a python file I created in the same path as this file, and it recorded database password,
 so you should create a same file like that, and make a method to return your password.
 I just don't wanna pull this file to Github.
 private_settings.py should like following:
 
    # coding=utf-8
    # Code by SmallSquare, 2020/5.
    # To save some private settings.
    
    def getMysqlPassword():
        return "123456"
        
"""


def insert_movie(movielist):
    db = pymysql.connect("localhost", "root", private_settings.getMysqlPassword(), "businessdataanalysis",
                         charset='utf8')

    # use method cursor() to get a 游标.
    cursor = db.cursor()

    sql = "INSERT INTO movie(id, title, rate) VALUES (%s, %s, %s)"

    try:
        for movie in movielist:
            cursor.execute(sql, (movie['id'], movie['title'], movie['rate']))
        db.commit()
    except Exception as e:
        # rollback when get error
        db.rollback()
        print("Insert ERROR, so rollback.")
        print(e)

    db.close()


def insert_comment(commentlist, movie_id):
    db = pymysql.connect("localhost", "root", private_settings.getMysqlPassword(), "businessdataanalysis",
                         charset='utf8mb4')
    # Must be 'utf8mb4' to be compatible to the 4个编码的 character.

    # use method cursor() to get a 游标.
    cursor = db.cursor()

    sql = "INSERT INTO comment(text, movie_id) VALUES (%s, %s)"

    try:
        for comment in commentlist:
            cursor.execute(sql, (comment, movie_id))
            print(comment)
        db.commit()
    except Exception as e:
        # rollback when get error
        db.rollback()
        print("Insert ERROR, so rollback.")
        print(e)

    db.close()


def del_all(table):
    db = pymysql.connect("localhost", "root", private_settings.getMysqlPassword(), "businessdataanalysis",
                         charset='utf8')

    # use method cursor() to get a 游标.
    cursor = db.cursor()

    sql = "DELETE FROM " + table

    try:
        cursor.execute(sql)
        db.commit()
    except Exception as e:
        # rollback when get error
        db.rollback()
        print("Delete ERROR, so rollback.")
        print(e)

    db.close()
