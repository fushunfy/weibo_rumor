from backend.User import User
from backend.Files import Files
from django.contrib import messages
from django.utils.datetime_safe import datetime
from django.conf import settings
from django.core.mail import send_mail
from random import Random
from django.db.models.signals import pre_delete
from django.dispatch.dispatcher import receiver
# from django.views.decorators.csrf import csrf_exempt,csrf_protect
from django.http import JsonResponse
import json
import codecs
import csv
import os
import re
import pandas as pd
# codecs专门用作编码转换，当我们要做编码转换的时候可以借助codecs很简单的进行编码转换
from requests_toolbelt.multipart.decoder import MultipartDecoder

User.test_database()
files = Files()


def random_password(random_length=6):
    temp_password = ''
    chars = 'abcdefghijklmnopqrstuvwsyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
    length = len(chars) - 1
    random = Random()
    for i in range(random_length):
        temp_password += chars[random.randint(0, length)]
    return temp_password


# send_mail的参数分别是：邮件标题，邮件内容，发件箱（settings.py中设置过的那个），收件箱列表（可以发送给多个人）
# 失败静默（若发送失败，报错提示我们）

def changepassword(request):
    if request.method == 'POST':
        username = request.POST['username']
        old_password = request.POST['old_password']
        new_password = request.POST['new_password']
        res = User.change_password(user_name=username, old_password=old_password, new_password=new_password)

        return JsonResponse(res, safe=False)


def islogining(request):
    if request.session['is_login'] == False:
        return False
    else:
        return True


# @csrf_exempt
def login(request):
    if request.method == 'POST':
        userinfo = json.loads(request.body)
        username = userinfo['username']
        password = userinfo['password']
        res = User.login(user_name=username, password=password)
        return JsonResponse(res, safe=False)


def register(request):
    if request.method == 'POST':
        userinfo = json.loads(request.body)
        username = userinfo['username']
        email = userinfo['email']
        password = userinfo['password']
        res = User.register(user_name=username, password=password, email=email)

        return JsonResponse(res, safe=False)


def logout(request):
    del request.session['username']
    request.session['is_login'] = False
    # messages.success(request, '已登出')
    # return HttpResponseRedirect('/')


def uploadFile(request):
    if request.method == 'POST':
        # headers = json.loads(json.dumps(dict(request.headers)))
        # contentType = headers['Content-Type']
        # fileInfo = MultipartDecoder(request.body, content_type=contentType)
        # print('ceshi')
        # print(fileInfo)
        # sys.exit()
        # files.setFileName(filename)

        # res = MultipartDecoder.from_response(request.body)
        """（1）在保存成CVS文件的时候，如果里面含有CVS格式不支持的字符的进候，Excel会提示出来，确定保存后，如果里面有,号，
        Excel会自动增加双引号，即","，以这种方式做库CVS中的数据，如,abc是一个数据，存储后会是",abc"，
        （2）所以你要先以"做为处理条件，将""两号之间的,号换成其他特殊字符，之后再以split用,号拆分"""


        fileInfo = request.body.decode('utf-8')

        fileInfoList = fileInfo.split('\r\n')
        context = fileInfoList[4]

        # context = context.replace({r'\s+$': '', r'^\s+': ''}, regex=True)
        # decode('utf-8')
        path = os.path.abspath('.')
        with codecs.open(path+'/backend/resources/data/test.csv', 'wb+', 'utf-8') as file_csv:
        # QUOTE_MINIMAL: 指示writer对象仅引用那些包含特殊字符（例如定界符，quotechar或lineterminator中的任何字符）的 字段。
        # dialect就是定义一下文件的类型，我们定义为csv类型,"newline="就是说因为我们的csv文件的类型，如果不加这个东西，当我们写入东西的时候，就会出现空行
            writer = csv.writer(file_csv, delimiter=',')
            contextList = context.split('\n')
            for item in contextList:
                if item != '':
                    while True:
                        index_range_obj = re.search(r'\"[^\"]*[^\"]*\"', item)
                        if index_range_obj:
                            index_range = index_range_obj.span()
                            item = item[0:index_range[0]]+item[index_range[0]+1:index_range[1]-1].replace(',', '，') + item[index_range[1]:]
                        else:
                            break
                    csvLine = item.split(',')
                    writer.writerow(csvLine)
            print("保存文件成功，处理结束")
        res = {'code': 0, 'message': 'uploadFile successfully!'}
        return JsonResponse(res, safe=False)


def runPredictFile(request):
    out = os.popen("ps -ef | grep predict_test.py | grep -v grep").readlines()
    if out:
        pid = out[0].split()[1]
        os.system('kill -9 ' + pid)
    cur_path = os.path.abspath('.')
    file_path = cur_path+'/backend/resources/predict_test.py'
    log_path = cur_path+'/backend/resources/predictlog.log'
    os.system('nohup python ' + file_path + ' > ' + log_path +' 2>&1 & ')
    res = {'code': 0, 'message': 'start predict!'}
    return JsonResponse(res, safe=False)


# def home(request):
#     if not islogining(request):
#         return HttpResponseRedirect('/')
#     else:
#         # blog_info = Blog_Info.objects.filter(user=request.session['user_id'])
#         username = request.session['username']
#         return render(request, 'home.html', locals())

# def uploadFile(request):
#     username = request.session['username']
#     if request.method == 'POST':
#         load_file = request.FILES['load_file']
#         if load_file:
#             selfIntroduction.head_image.delete(False)
#             selfIntroduction.head_image=head_image
#             selfIntroduction.save()
#     return HttpResponseRedirect('/personalinfo/')
