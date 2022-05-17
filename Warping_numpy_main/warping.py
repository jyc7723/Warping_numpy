import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from numpy.linalg import inv    #주어진 행렬의 역행렬계산
import math
import warnings

warnings.filterwarnings(action='ignore') #경고 메세지 끄기

img = Image.open("img.jpg").convert('L') #이미지 불러오기
img = np.array(img) #이미지배열 생성

img2 = Image.open("img2.jpg").convert('L') #이미지 불러오기
img2 = np.array(img2) #이미지배열 생성

global zeros_bg #전역함수처리
zeros_bg = np.zeros_like(img) #같은 사이즈의 zeros 만들기

def linear(x1,y1,x2,y2):
    global X,Y
    x = np.array(range(600))  
    y = np.array(range(800))
    X,Y = np.meshgrid(x,y) #그리드 만들기
    z = ((y2 - y1)/(x2 - x1))*(X - x1)+(y1 - Y) #직선의 방정식
    gradient = (y2 - y1)/(x2 - x1) #기울기
    return z, gradient

def dot_choice(image,image2): #점찍기 함수
    plt.imshow(image,'gray')
    print ( "Please click img1" )
    dot1 = (np.around(plt.ginput(20)).T).astype(np.int) #좌표 식별
    print(dot1)
    plt.imshow(image2,'gray')
    print ( "Please click img2" )
    dot2 = (np.around(plt.ginput(20)).T).astype(np.int) #좌표 식별 
    print(dot2)
    
    return dot1, dot2

def area_choice(dot):
    w1, h1 = linear(dot[0][0],dot[1][0],dot[0][1],dot[1][1]) 
    w2, h2 = linear(dot[0][0],dot[1][0],dot[0][2],dot[1][2])
    w3, h3 = linear(dot[0][3],dot[1][3],dot[0][2],dot[1][2])
    w4, h4 = linear(dot[0][3],dot[1][3],dot[0][1],dot[1][1])
    if h2<0 and h4<0:
        area = np.where((w1<0) & (w2<0) & (w3>0) & (w4>0))
        return area
    elif h2<0:
        area = np.where((w1<0) & (w2<0) & (w3>0) & (w4<0))
        return area
    elif h4<0:
        area = np.where((w1<0) & (w2>0) & (w3>0) & (w4>0))
        return area
    else:
        area = np.where((w1<0) & (w2>0) & (w3>0) & (w4<0))
        return area

def warping(dot1,dot2,image_vec):
    x = dot1[0]
    y = dot1[1]
    xy = x*y
    n1 = np.ones((1,np.size(x)))
    xy_vstack = np.vstack((xy,x,y,n1)) #세로결합
    xy_pinv = np.linalg.pinv(xy_vstack) #역행렬구하는 함수
    change = np.dot(dot2,xy_pinv)

    xx = image_vec[1]
    yy = image_vec[0]
    xxyy = xx*yy
    n2 = np.ones((1,np.size(xx)))
    xxyy_vstack = np.vstack((xxyy,xx,yy,n2))

    war_picture_vec = np.int_(np.dot(change,xxyy_vstack))
    zeros_bg[war_picture_vec[1],war_picture_vec[0]] = img[image_vec[0],image_vec[1]]

    return zeros_bg

########################## main ##############################
    
img_dot,img2_dot = dot_choice(img,img2)

for i in range(0,15):
    if i+1 % 3 == 0 and i != 0:
            continue
    img_dot_list = np.array([[img_dot[0][i], img_dot[0][i+1], img_dot[0][i+4], img_dot[0][i+5]],
                        [img_dot[1][i], img_dot[1][i+1], img_dot[1][i+4], img_dot[1][i+5]]])
    
    img2_dot_list = np.array([[img2_dot[0][i], img2_dot[0][i+1], img2_dot[0][i+4], img2_dot[0][i+5]],
                        [img2_dot[1][i], img2_dot[1][i+1], img2_dot[1][i+4], img2_dot[1][i+5]]])

    choice_vec = np.array(area_choice(img_dot_list))                         
    zeros = warping(img_dot_list,img2_dot_list, choice_vec)

plt.imshow(zeros,'gray')
plt.show()








