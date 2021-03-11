from flask import Flask, request
from flask_cors import CORS
import json
import numpy
import re

# 设定一个应用程序
app = Flask(__name__)

# r'/*' 是通配符，让本服务器所有的 URL 都允许跨域请求
CORS(app, resources=r'/*')

# 只接受POST方法访问
@app.route("/test", methods=["POST"])
def check():
    # 默认返回内容
    return_dict = {'return_code': '200', 'return_info': '处理成功', 'lat': False, 'lng': False, 'i':False}
    # 判断传入的json数据是否为空
    if request.get_data() is None:
        return_dict['return_code'] = '5004'
        return_dict['return_info'] = '请求参数为空'
        return json.dumps(return_dict, ensure_ascii=False)
    #
    data = numpy.loadtxt(open("E:\\PyCharm\\PycharmProjects\\differential-privacy-trajectory-data-publication\\cluster_centers.csv", "rb"), delimiter=",", skiprows=0)
    data = numpy.around(data, decimals=6)
    # 获取传入的参数
    get_Data = request.get_data()
    # 传入的参数为bytes类型，需要转化成json
    get_Data = json.loads(get_Data)
    pstring = get_Data.get('P')
    if(pstring!=''):
        plist = re.findall(r'\d+',pstring)
        p = int(plist[0])-1
        lstring = get_Data.get('L')
        llist = re.findall(r'\d+', lstring)
        l = int(llist[0])
        # 对参数进行操作
        return_dict['lat'] = data[l,2*p]
        return_dict['lng'] = data[l,2*p+1]
    else:
        lstring = get_Data.get('L')
        llist = re.findall(r'\d+', lstring)
        l = int(llist[0])
        lat = data[l,0].astype(str)
        lng = data[l,1].astype(str)
        for i in range(1,10):
            l = int(llist[i])
            lattemp = data[l,2*i].astype(str)
            lngtemp = data[l,2*i+1].astype(str)
            lat = lat + ","+ lattemp
            lng = lng + ","+ lngtemp
        return_dict['lat'] = lat
        return_dict['lng'] = lng
        return_dict['i'] = i

    return json.dumps(return_dict, ensure_ascii=False)

if __name__ == "__main__":
    app.run(debug=True)