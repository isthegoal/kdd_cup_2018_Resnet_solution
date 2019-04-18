'''
根据模板合并数据
beijing_17_18_aq.csv
beijing_17_18_meo.csv
模板
merage_aq_meo_beijing_template.csv
输出文件
assets/output/merage_aq_meo/merage_aq_meo_beijing.csv
'''

from math import*

'''
计算经纬度之间的距离方法一
@Lat  is latitude 维度
@Lng is longitude 经度
'''


def Distance1(Lat_A,Lng_A,Lat_B,Lng_B): #第一种计算方法
    ra=6378.140 #赤道半径
    rb=6356.755 #极半径 （km）
    flatten=(ra-rb)/ra  #地球偏率
    rad_lat_A=radians(Lat_A)
    rad_lng_A=radians(Lng_A)
    rad_lat_B=radians(Lat_B)
    rad_lng_B=radians(Lng_B)
    pA=atan(rb/ra*tan(rad_lat_A))
    pB=atan(rb/ra*tan(rad_lat_B))
    xx=acos(sin(pA)*sin(pB)+cos(pA)*cos(pB)*cos(rad_lng_A-rad_lng_B))
    c1=(sin(xx)-xx)*(sin(pA)+sin(pB))**2/cos(xx/2)**2
    c2=(sin(xx)+xx)*(sin(pA)-sin(pB))**2/sin(xx/2)**2
    dr=flatten/8*(c1-c2)
    distance=ra*(xx+dr)
    return distance


'''
计算经纬度之间的距离方法二
@Lat  is latitude 维度
@Lng is longitude 经度
'''
def Distance2(lat1,lng1,lat2,lng2):# 第二种计算方法
    radlat1=radians(lat1)
    radlat2=radians(lat2)
    a=radlat1-radlat2
    b=radians(lng1)-radians(lng2)
    s=2*asin(sqrt(pow(sin(a/2),2)+cos(radlat1)*cos(radlat2)*pow(sin(b/2),2)))
    earth_radius=6378.137
    s=s*earth_radius
    if s<0:
        return -s
    else:
        return s