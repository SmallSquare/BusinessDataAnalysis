# coding=utf-8

# Code by SmallSquare, 2020/5.
# Provide an easy access to Mysql.
import sys

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

    sql = "INSERT INTO comment(text, rate , movie_id) VALUES (%s, %s, %s)"

    try:
        for comment in commentlist:
            cursor.execute(sql, (comment["comment"], comment["star"], movie_id))
            print(comment)
        db.commit()
    except Exception as e:
        # rollback when get error
        db.rollback()
        print("Insert ERROR, so rollback.")
        print(e)
        print(sys.exc_info())

    db.close()


def get_movies():
    movie_list = []

    db = pymysql.connect("localhost", "root", private_settings.getMysqlPassword(), "businessdataanalysis",
                         charset='utf8')
    cursor = db.cursor()

    sql = "SELECT * FROM movie"
    try:
        cursor.execute(sql)
        results = cursor.fetchall()
        for row in results:
            id = row[0]
            title = row[1]
            rate = row[2]
            # print("id=%s,title=%s,rate=%s" % (id, title, rate))
            movie_list.append({"id": id, "title": title, "rate": rate})

    except Exception as e:
        print("Unable to fetch data.")

    db.close()

    return movie_list


def get_comments(movie_id):
    comment_list = []

    db = pymysql.connect("localhost", "root", private_settings.getMysqlPassword(), "businessdataanalysis",
                         charset='utf8mb4')
    cursor = db.cursor()

    sql = "SELECT * FROM comment WHERE movie_id = " + str(movie_id)
    try:
        cursor.execute(sql)
        results = cursor.fetchall()
        for row in results:
            id = row[0]
            text = row[1]
            movie_id = row[2]
            rate = row[3]
            # print("id=%s,text=%s,movie_id=%s" % (id, text, movie_id))
            comment_list.append({"id": id, "text": text, "movie_id": movie_id, "star": rate})
    except Exception as e:
        print("Unable to fetch data.")

    db.close()

    return comment_list


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
