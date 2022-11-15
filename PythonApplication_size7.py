from tkinter import Image
from scipy.spatial import distance as dist
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from imutils import perspective
from imutils import contours
from collections import Counter
from time import sleep
import numpy as np
import imutils
import cv2
import statistics as stat

def midpoint(ptA, ptB):
    return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)

def A2RGB2(color):
    a = [int(color[0]), int(color[1]), int(color[2])]
    return a

def get_colors(image, number_of_colors):
  
    clf = KMeans(n_clusters = number_of_colors)
    labels = clf.fit_predict(image)
    
    counts = Counter(labels)
    #counts = dict(sorted(counts.items()))
    counts = dict(counts.items())
    
    center_colors = clf.cluster_centers_
    ordered_colors = [center_colors[i] for i in counts.keys()]
    result0=str(round (counts[0]/(counts[0]+counts[1])*100, 2))
    result1=str(round (counts[1]/(counts[0]+counts[1])*100, 2))
    col0 = A2RGB2(ordered_colors[0])

    #if (col0[0]<20 and col0[1]<20 and col0[2]<20):
    #    col1=[0,0,0]
    col1 = A2RGB2(ordered_colors[1])

    #if (col1[0]>=20 and col1[1]>=20 and col1[2]>=20):
    #    col1=[255,255,255]
    arr=((result0,col0), (result1, col1))
    return arr #rgb_colors

cap = cv2.VideoCapture("test6.mp4")

# Переменные для обрезки экрана:
startX = 10
stopX = 360
startY = 10
stopY = 640

f_width = 640 #cap.get(3)  # float берем размеры кадра
f_height = 360 #cap.get(4) # float берем размеры кадра

ccount = 0
counter=1
kernel1 = np.ones((5, 5), 'uint8') #матрица свёртки изоюражения для dilate

black_array = [0]
white_array = [0]
c = [0]  # массив для суммы значений цветов приотображении графиков
median_black=[0]
i=0

while cap.isOpened():
    ret, image = cap.read()
    modified_image = cv2.resize(image, (640, 360), interpolation = cv2.INTER_AREA)
    modified_image2 = modified_image[startX:stopX, startY:stopY]         # Обрезаем изображение, сперва высота, потом ширина
    
    hsv = cv2.cvtColor(modified_image2, cv2.COLOR_BGR2HSV)
    hsv = cv2.GaussianBlur(hsv, (3, 3), 0)
    #hsv = cv2.bilateralFilter(hsv,5,10,10)
    #hsv = cv2.medianBlur(hsv,3)
  
    #lower_red = np.array([10,50,30])        # рабочий вариант для
    #upper_red = np.array([20,70,100])       # SCHOM2.mp4
    #lower_red = np.array([10,0,90])   # рабочий вариант для
    #upper_red = np.array([50,30,130]) # SCHOM.mp4
    lower_red = np.array([10,30,150])   # рабочий вариант для
    upper_red = np.array([30,200,250]) # test6.mp4
    mask = cv2.inRange(hsv, lower_red, upper_red)
    res = cv2.bitwise_and(modified_image2,modified_image2, mask= mask)
    
    
    edged = cv2.Canny(res, 50, 100)
    edged = cv2.dilate(edged, kernel1, cv2.BORDER_REFLECT, iterations=2)
    
    crop_image = edged
    test = get_colors(crop_image, 2)
    #cv2.circle(edged, (60,60), 55, (0,0,255), -1)  ##
    #cv2.putText(edged, test[0][0], (18,55), cv2.FONT_HERSHEY_SIMPLEX, 1, test[0][1], 2) ##
    #cv2.putText(edged, test[1][0], (18,80), cv2.FONT_HERSHEY_SIMPLEX, 1, test[1][1], 2) ##
    black_array.append(float(test[0][0])) 
    white_array.append(float(test[1][0]))
    i+=1
    #cv2.putText(crop_image, test, (30,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
    cv2.imshow("Image", edged) ##
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()

# Делаем массив медианых значений темного цвета
#for k in range (0, len(black_array), 10):
#    median_black.append(stat.median(black_array[k:k+10]))
#x_median=np.arange(1, (len(median_black))*10, 10)

for k in range (0, len(white_array)):
    median_black.append(stat.median(white_array[k:k+20]))
x_median=np.arange(1, len(median_black)+1)

# делаем суммарный массив - складываем проценты цветов
for k in range(len(white_array)):
    c.append(float(black_array[k])+float(white_array[k]))
x=np.arange(1, len(black_array)+1)
xc=np.arange(1, len(c)+1)

# рисуем графики
plt.title("Colors")
plt.xlabel("Iteration")
plt.ylabel("Colors percentage")
#plt.plot(x, black_array, x, white_array, x_median, median_black)
plt.plot(x_median, median_black)
#plt.plot(x, black_array, x, white_array, xc, c)
plt.show()


