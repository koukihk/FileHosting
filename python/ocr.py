# coding=utf-8

import sys
import json
import base64


# 保证兼容python2以及python3
IS_PY3 = sys.version_info.major == 3
if IS_PY3:
    from urllib.request import urlopen
    from urllib.request import Request
    from urllib.error import URLError
    from urllib.parse import urlencode
    from urllib.parse import quote_plus
else:
    import urllib2
    from urllib import quote_plus
    from urllib2 import urlopen
    from urllib2 import Request
    from urllib2 import URLError
    from urllib import urlencode

# 防止https证书校验不正确
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

# 修改为你的个人配置
API_KEY = 'GAa7DwzDRBppn1bIjdtX3TKO'
SECRET_KEY = 'AcCFaa8BplfAGMZhw2vDCkHtGGrLbpH0'
# 通用文字识别
OCR_URL1 = "https://aip.baidubce.com/rest/2.0/ocr/v1/accurate_basic"
# 车牌识别
OCR_URL2 = "https://aip.baidubce.com/rest/2.0/ocr/v1/license_plate"


"""  TOKEN start """
TOKEN_URL = 'https://aip.baidubce.com/oauth/2.0/token'


"""
    获取token
"""
def fetch_token():
    params = {'grant_type': 'client_credentials',
              'client_id': API_KEY,
              'client_secret': SECRET_KEY}
    post_data = urlencode(params)
    if (IS_PY3):
        post_data = post_data.encode('utf-8')
    req = Request(TOKEN_URL, post_data)
    try:
        f = urlopen(req, timeout=5)
        result_str = f.read()
    except URLError as err:
        print(err)
    if (IS_PY3):
        result_str = result_str.decode()


    result = json.loads(result_str)

    if ('access_token' in result.keys() and 'scope' in result.keys()):
        if not 'brain_all_scope' in result['scope'].split(' '):
            print ('please ensure has check the  ability')
            exit()
        return result['access_token']
    else:
        print ('please overwrite the correct API_KEY and SECRET_KEY')
        exit()

"""
    读取文件
"""
def read_file(image_path):
    f = None
    try:
        f = open(image_path, 'rb')
        return f.read()
    except:
        print('read image file fail')
        return None
    finally:
        if f:
            f.close()


"""
    调用远程服务
"""
def request(url, data):
    req = Request(url, data.encode('utf-8'))
    has_error = False
    try:
        f = urlopen(req)
        result_str = f.read()
        if (IS_PY3):
            result_str = result_str.decode()
        return result_str
    except  URLError as err:
        print(err)

if __name__ == '__main__':

    # 获取access token
    token = fetch_token()

    # 拼接通用文字识别高精度url
    image_url1 = OCR_URL1 + "?access_token=" + token
    # 拼接车牌识别高精度url
    image_url2 = OCR_URL2 + "?access_token=" + token

    text = ""

    # 读取测试图片
    file_content = read_file('img/LPR/card_img0.jpg')

    # 调用文字识别服务
    result1 = request(image_url1, urlencode({'image': base64.b64encode(file_content)}))
    # 调用车牌识别服务
    result2 = request(image_url2, urlencode({'image': base64.b64encode(file_content)}))

    # 解析返回结果
    result_json1 = json.loads(result1)
    print(result1)
    result_json2 = json.loads(result2)
    print(result2)

    # 通用文字识别
    for words_result in result_json1["words_result"]:
        text = text + words_result["words"]
    # 打印文字
    print('通用文字识别结果：')
    print(text)

    text = ""

    # 车牌识别
    text = text + result_json2["words_result"]["number"]
    print('车牌识别结果：')
    print(text)

