from django.db import connection
import os

class Files:  # 写空调开关数据库

    filename = ''

    def setFileName(self, filename):
        if os.path.exists('resources/data/'+filename):
            os.remove('resources/data/'+filename)
        self.filename = filename


    @staticmethod
    def test_database():
        sql = "select * from LoginInfo"
        with connection.cursor() as cursor:
            cursor.execute(sql)
            user_info = cursor.fetchall()

        if user_info:
            print('connect database successfully!')
        else:
            print('connect database unsuccessfully!')

    @staticmethod
    def login(user_name, password):

        sql = "select * from LoginInfo where username = '%s'" % user_name
        with connection.cursor() as cursor:
            cursor.execute(sql)
            user_info = cursor.fetchone()

        if user_info:
            if password == user_info[1]:
                res = {'code': 0, 'message': 'login successfully!'}
            else:
                res = {'code': 400, 'message': 'password is not right!'}
        else:
            res = {'code': 400, 'message': 'username is not exist!'}
        print(res)
        return res

    @staticmethod
    def register(user_name, password, email):
        sql = "select * from LoginInfo where username='%s'" % user_name
        with connection.cursor() as cursor:
            cursor.execute(sql)
            user_info = cursor.fetchone()

        if user_info:
            res = {'code': 400, 'message': 'username already exists!'}
        else:
            sql = "insert into LoginInfo (username, password, email) values ('%s', '%s', '%s')" % user_name
            with connection.cursor() as cursor:
                cursor.execute(sql)
            res = {'code': 0, 'message': 'register successfully!'}
        print(res)
        return res

    @staticmethod
    def change_password(user_name, old_password, new_password):
        sql = "select * from LoginInfo where username='%s'" % user_name
        with connection.cursor() as cursor:
            cursor.execute(sql)
            user_info = cursor.fetchone()

        if user_info:
            print(type(user_info))
            print(user_info[1])
            if old_password == user_info[1]:
                sql = "update LoginInfo set password = '%s' where username = '%s'" % (new_password, user_name)
                with connection.cursor() as cursor:
                    cursor.execute(sql)
                res = {'code': 0, 'message': 'password is rights!'}
            else:
                res = {'code': 400, 'message': 'password is not right!'}
        else:
            res = {'code': 400, 'message': 'username is not exist!'}
        print(res)
        return res
