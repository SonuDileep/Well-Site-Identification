import sys 
import time
sys.path.insert(0,'/mnt/Data/YOLO_NEW/darknet-master/')
from darknet_images_equip import main2 
from darknet_images_equip import dettect
import numpy as np
import math  
import json
import qimage2ndarray
import cv2

###### TO FIND THE WELL HEADS INSIDE THE RECTANGULAR AREA
from qgis.core import QgsProject
import math

cfg = '/mnt/Data/YOLO_NEW/darknet-master/complete_eq/yolov4(2).cfg'
data = '/mnt/Data/YOLO_NEW/darknet-master/complete_eq/detector.data'
weights = '/mnt/Data/YOLO_NEW/darknet-master/complete_eq/yolov4_21000.weights'
net,cn,cc = main2(cfg,data,weights)

layers_list = [] ## to store all bounding box coordinates

def if_in_map_canvas(x1, y1,cord):
    if x1 >= cord[0] and y1 <= cord[1] and x1 <= cord[4] and y1 >= cord[5]:
        return True
    else:
        return False
        
def calc_dist(x1, y1, x2, y2):
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

def return_points(poly,xform):
    layer = QgsProject.instance().mapLayersByName('Wells')

    layer_1 = QgsProject.instance().mapLayersByName('DJStudyAreaBoundary')
    ext = layer_1[0].extent()
    cord_x1 = ext.xMinimum()
    cord_x2 = ext.xMaximum() 
    cord_y1 = ext.yMaximum() 
    cord_y2 = ext.yMinimum() 
    cord = [cord_x1,cord_y1,cord_x2,cord_y1,cord_x2,cord_y2,cord_x1,cord_y2]
    i = 0
    a = []
    p = []
    t = 0
    print("Hre")
    for f in layer[0].getFeatures():
        geom = f.geometry()
        if poly.geometry().contains(QgsPointXY(xform.transform(QgsPointXY(geom.asPoint().x(), geom.asPoint().y())))) == True:

            a.append((geom.asPoint().x(), geom.asPoint().y()))
    #       print(geom.asPoint().x(),geom.asPoint().y())
    print("Done")
    i = 0
    k = 1

    l = len(a)
    print(l)
    if l < 2:
        return a
    while i < l - 1 and len(a) >= 2:
        num1 = a[i]
        k = i + 1
        while k < l:
            num2 = a[k]
            dist = calc_dist(num1[0], num1[1], num2[0], num2[1])
            if dist < 60:
                a[i] = (((num1[0] + num2[0]) / 2, (num1[1] + num2[1]) / 2))
                a.pop(k)
                i = i - 1
                l = len(a)
                break
            k = k + 1
        i = i + 1

    return a

#####################

###################### Basic Core Functions #####################
def detect(img,threshold): # for detection using YOLO
    res = dettect(img, net, cn,cc, threshold)
    return res

def get_image(coord,i,n,mm): #saves the image using coord and i is the number of images
# n=False normally.set it true when dealing with zoomed out images of well apd    
    if len(coord)==0:
        return 
    w = 606
    h = 606
    if n==True:
        w = (coord[2] - coord[0]) * 6
        h = (coord[1] - coord[3]) * 6
    img = QImage(QSize(int(w), int(h)), QImage.Format_ARGB32_Premultiplied)
 
# set background color
    color = QColor(255, 255, 255)
    img.fill(color.rgb())
 
# create painter
    p = QPainter()
    p.begin(img)
    p.setRenderHint(QPainter.Antialiasing)
 
# create map settings
    ms = QgsMapSettings()
    ms.setBackgroundColor(color)
 
# set layers to render
    layer = QgsProject.instance().mapLayersByName('google_satellite')
    ms.setLayers([layer[0]])

    rect = QgsRectangle(coord[0],coord[1],coord[2],coord[3])
#rect.scale(1.1)
    ms.setExtent(rect)
 
# set ouptut size
    ms.setOutputSize(img.size())
 
## setup qgis map renderer
    render = QgsMapRendererCustomPainterJob(ms, p)
    render.start()
    render.waitForFinished()
    p.end()
    img_loc = qimage2ndarray.rgb_view(img)
    img_loc = cv2.cvtColor(img_loc, cv2.COLOR_BGR2RGB)
    return img_loc

def draw_layer(new_x1,new_y1,new_x2,new_y2,name):
    rect =  QgsRectangle(new_x1,new_y1,new_x2,new_y2)
    rectangleLayer = QgsVectorLayer("Polygon?crs=EPSG:3857", name, "memory")
    rectangleLayerProvider = rectangleLayer.dataProvider()
    newFeat = QgsFeature()
    symbol = QgsFillSymbol.createSimple({'name': 'square', 'color':'255,0,255,0', 'width_border':'0.8'})
    geom = QgsGeometry.fromWkt(rect.asWktPolygon())
    newFeat.setGeometry(geom)
    rectangleLayerProvider.addFeatures([newFeat])
    QgsProject.instance().addMapLayer(rectangleLayer)
    rectangleLayer.renderer().setSymbol(symbol) 
    
def draw_range(cord): # draws the rectangular search space ( 4 input points from user)
    layer = QgsVectorLayer('Polygon?crs=EPSG:3857', 'area', 'memory'
                           )
    pr = layer.dataProvider()
    poly = QgsFeature()
    points = [QgsPointXY(cord[0], cord[1]),
              QgsPointXY(cord[2], cord[3]),
              QgsPointXY(cord[4], cord[5]),
              QgsPointXY(cord[6], cord[7])]
    poly.setGeometry(QgsGeometry.fromPolygonXY([points]))
    pr.addFeatures([poly])
    layer.updateExtents()
    symbol = QgsFillSymbol.createSimple({'name': 'square',
            'color': '255,0,255,0', 'width_border': '0.8'})
    symbol.symbolLayer(0).setStrokeColor(QColor(0, 0, 255))
    QgsProject.instance().addMapLayers([layer])
    layer.renderer().setSymbol(symbol) 
    return poly 

    
########################### Functions which handles equipment detections #########################
def check_intersection(a,b):
    m = True
    if a[0] >= b[2] or b[0] >= a[2]:
        m = False
    if a[1] <= b[3] or b[1] <= a[3]:
        m = False
    if m == False:
        return False,[],[]
    w1 = abs(a[2] - a[0])
    h1 = abs(a[3] - a[1])
    w2 = abs(b[2] - b[0])
    h2 = abs(b[3] - b[1])
    a1 = w1 * h1
    a2 = w2 * h2

    dx = min(a[2], b[2]) - max(a[0], b[0])
    dy = min(a[1], b[1]) - max(a[3], b[3])
    overlap = abs(dx*dy)
    if (dx >= 0) and (dy >= 0):
        if a1 >= a2:  # means a1 is the larger rectangle
            diff = (overlap * 100) / a2
            if diff > 60:
                p = combine(a, b)
                return True, p, []
            else:
                return False, a, b
        if a2 > a1:  # means a2 is the larger rectangle
            diff = (overlap * 100) / a1
            if diff > 60:
                p = combine(a, b)
                return True, p, []
            else:
                return False, a, b
    else:  # case when 2 rectangles does not intersect
        return False, a, b

def combine(a1, a2):
    x_min3 = min(a1[0], a2[0])
    x_max3 = max(a1[2], a2[2])
    y_min3 = max(a1[1], a2[1])
    y_max3 = min(a1[3], a2[3])
    return [x_min3, y_min3, x_max3, y_max3]


def equip_clean(temp):
    length = len(temp)
    if length < 2:
        return temp
    i = 0
    while i < (length - 1) and len(temp)>=2:
        j = i + 1
        while j < length :
            t2, a1,a2 = check_intersection(temp[i], temp[j])
            if t2 == True:
                temp[i] = a1
                temp.pop(j)
                i = i - 1
                length = len(temp)
                break
            j = j + 1
        i = i + 1
    return temp

def check_equip(loc_coord,m): # to check welltank inside wellpad
    x1_equip = loc_coord[0]
    y1_equip = loc_coord[1]
    x2_equip = loc_coord[2]
    y2_equip = loc_coord[3]
    add_1 = 105
    add_2 = 85
    x1 = x1_equip
    x2 = x1 + add_1
    tank_list = []
    while x1 <= x2_equip:
        y1 = y1_equip
        y2 = y1 - add_1
        flag = 0
        while y1 >= y2_equip:
            img_loc1 = get_image([x1,y1,x2,y2],1,True,False)
            gray = cv2.cvtColor(img_loc1, cv2.COLOR_RGB2GRAY)
            no = np.sum(gray == 255)
            if flag == 40:
                y1 = y1 - add_2
                y2 = y2 - add_2
                flag = 0    
                continue
            if no > 10000:
                flag = flag + 1
                continue
            else:
                flag = 0
            c = detect_tanks(img_loc1,0.6)
            if c!=False:
                d1 = abs(x2 - x1)
                d2 = abs(y1 - y2)
                w = d1 * 6
                h = d2 * 6
#                g = []
                for i in range(len(c)):
                    if c[i][0] == 'well_tank':
                        x11, y11, w_size, h_size = c[i][2]
                        x_start = round(x11 - (w_size/2))
                        y_start = round(y11 - (h_size/2))
                        x_end = x_start + w_size
                        y_end = y_start + h_size
                        new_x1 = x1 + (d1/w) * x_start
                        new_x2 = x1 + (d1/w) * x_end
                        new_y1 = y1 - (d2/h) * y_start
                        new_y2 = y1 - (d2/h) * y_end
                        if new_x1 < x2_equip and new_x2 < x2_equip and new_y1 > y2_equip and new_y2 > y2_equip:
                            tank_list.append([new_x1,new_y1,new_x2,new_y2])
#                if len(g)!=0:
#                    tank_list.extend(g)
                
            y1 = y1 - add_2
            y2 = y2 - add_2
        x1 = x1 + add_2
        x2 = x2 + add_2
    tank_list = equip_clean(tank_list)  # Combines the detections with intersections  
    tank_list = equip_clean(tank_list)
    return tank_list

#########################################


####################### To help find the well pads ####################

def find_layer(c,img_loc,L,m,width,height):
    
    x1 = L[0]
    y1 = L[1]
    d1 = abs(L[2]-L[0])
    d2 = abs(L[3]-L[1])
    #d1 = cord.xMaximum() - cord.xMinimum()
    #d2 = cord.yMaximum() - cord.yMinimum()
    g =[]
    w=width
    h=height
    w = 606
    h = 606
    for i in range(len(c)):
        x11, y11, w_size, h_size = c[i][2]
        x_start = round(x11 - (w_size/2))
        y_start = round(y11 - (h_size/2))
        x_end = x_start + w_size
        y_end = y_start + h_size
        new_x1 = x1 + (d1/w) * x_start
        new_x2 = x1 + (d1/w) * x_end
        new_y1 = y1 - (d2/h) * y_start
        new_y2 = y1 - (d2/h) * y_end
        g.append([new_x1,new_y1,new_x2,new_y2])            
        draw_layer(new_x1,new_y1,new_x2,new_y2,c[i][0])            
#            draw_layer(L[0],L[1],L[2],L[3],'well_pad')    
#            check_equip([new_x1,new_y1,new_x2,new_y2],m)
    return g 
    
def draw_under_conditions(m,x1,x2,y1,y2):
    add_1 = 450
    add_n = 50
    for i in range(len(m)):
#        if m[i][0] <= x1+add_1 and  m[i][1] >= y1-add_1:
        layers_list.append(m[i])
            
        if (abs(x2 - m[i][2]) < add_n and abs(m[i][3] - y2) < add_n) :
            points = extended_search(m[i], x2, y2, x1, y1,1)
            for j in range(len(points)):
                if points[j][3] >=y_max and points[j][2]<=x_max and points[j][0]>=x_min and points[j][1]<=y_min:
                    layers_list.append(points[j])
            continue

        if (abs(x2 - m[i][2]) < add_n) :
            points = extended_search(m[i], x2, y2, x1, y1,2)
            for j in range(len(points)):
                if points[j][3] >=y_max and points[j][2]<=x_max and points[j][0]>=x_min and points[j][1]<=y_min:
                    layers_list.append(points[j])
            continue

        if (abs(m[i][3] - y2) < add_n) :
            points = extended_search(m[i], x2, y2, x1, y1,3)
            for j in range(len(points)):
                if points[j][3] >=y_max and points[j][2]<=x_max and points[j][0]>=x_min and points[j][1]<=y_min:
                    layers_list.append(points[j])
            continue
                
                
def extended_search(m,x2,y2,x1,y1,flag):
    add_1 = 450
    thresh = 30
    add_size = 70
    if flag==1:
        new_x1 = m[0] - 5
        new_y1 = m[1] + 5
        new_x2 = m[2] + add_size
        new_y2 = m[3] - add_size
        img_location = get_image([new_x1, new_y1, new_x2,
                new_y2], 1, False,True)
        cord = get_extended_coordinate(img_location,
                [new_x1, new_y1, new_x2, new_y2])
        new_temp = []
        for i in range(len(cord)):
#                if abs(cord[i][0] - m[0]) <= 10 and abs(cord[i][1]
#                           - m[1]) <= 10:
                new_temp.append(cord[i])
        return new_temp
    
    if flag==2:
        new_x1 = m[0] - 5
        new_y1 = m[1] + 5
        new_x2 = m[2] + add_size
        new_y2 = m[3] - 5
        img_location = get_image([new_x1, new_y1, new_x2,
                new_y2], 1, False,True)
        cord = get_extended_coordinate(img_location,
                [new_x1, new_y1, new_x2, new_y2])
        new_temp = []
        for i in range(len(cord)):
#                if abs(cord[i][0] - m[0]) <= 10 and abs(cord[i][1]
#                           - m[1]) <= 10:
                new_temp.append(cord[i])
        return new_temp
                
    if flag==3:
        new_x1 = m[0] - 5
        new_y1 = m[1] + 5
        new_x2 = m[2] + 5
        new_y2 = m[3] - add_size
        img_location = get_image([new_x1, new_y1, new_x2,
                new_y2], 1, False,True)
        cord = get_extended_coordinate(img_location,
                [new_x1, new_y1, new_x2, new_y2])
        new_temp = []
        for i in range(len(cord)):
#                if abs(cord[i][0] - m[0]) <= 10 and abs(cord[i][1]
#                           - m[1]) <= 10:
                new_temp.append(cord[i])
        return new_temp
        
def get_extended_coordinate(img_location,L):
    c = detect(img_location,0.5)
    x1 = L[0]
    y1 = L[1]
    d1 = abs(L[2]-L[0])
    d2 = abs(L[3]-L[1])
    g =[]
    w=606
    h=606
    if c!=False:
        for i in range(len(c)):
            if c[i][0] !='well_pad':
                x11, y11, w_size, h_size = c[i][2]
                x_start = round(x11 - (w_size/2))
                y_start = round(y11 - (h_size/2))
                x_end = x_start + w_size
                y_end = y_start + h_size
                new_x1 = x1 + (d1/w) * x_start
                new_x2 = x1 + (d1/w) * x_end
                new_y1 = y1 - (d2/h) * y_start
                new_y2 = y1 - (d2/h) * y_end
#                if new_x1<=x1+30 and new_y1 >=y1-30:                  
                if new_x1 < x_max and new_y1 > y_max:
                    g.append([new_x1,new_y1,new_x2,new_y2])
    return g

##################################################################
            
################## QGIS Layer Implementation #######################        
#def merge_layer(new_x1, new_y1, new_x2, new_y2, i):
#    rect = QgsRectangle(new_x1, new_y1, new_x2, new_y2)
#    rectangleLayer = QgsVectorLayer('Polygon?crs=EPSG:3857', 'well_tank',
#                                    'memory')
#    rectangleLayerProvider = rectangleLayer.dataProvider()
#    newFeat = QgsFeature()
#    symbol = QgsFillSymbol.createSimple({'name': 'square',
#                                         'color': '255,255,0,0', 'width_border': '0.8'})
#    symbol.symbolLayer(0).setStrokeColor(QColor(0, 255, 0))
#    geom = QgsGeometry.fromWkt(rect.asWktPolygon())
#    newFeat.setGeometry(geom)
#    rectangleLayerProvider.addFeatures([newFeat])
#    return rectangleLayer

def merge_layer(new_x1, new_y1, new_x2, new_y2, i):
    x = (new_x1 + new_x2)/2
    y = (new_y1 + new_y2)/2
    rect = QgsPointXY(x , y)
    m = [rect]
    rectangleLayer = QgsVectorLayer('Point?crs=EPSG:3857', 'well_tank',
                                    'memory')
    rectangleLayerProvider = rectangleLayer.dataProvider()
    newFeat = QgsFeature()
#    symbol = QgsFillSymbol.createSimple({'name': 'square',
#                                         'color': '255,255,0,0', 'width_border': '0.8'})
#    symbol.symbolLayer(0).setStrokeColor(QColor(0, 255, 0))
#    geom = QgsGeometry.fromPolygonXY([m])
    geom = QgsGeometry.fromWkt(rect.asWkt())
    newFeat.setGeometry(geom)
    rectangleLayerProvider.addFeatures([newFeat])
     
    return rectangleLayer

def addlayer_from_list(list_name,name):
    memory = 'memory:' + name
    l1 = []
    for i in range(len(list_name)):
        l1.append(merge_layer(list_name[i][0], list_name[i][1],list_name[i][2],list_name[i][3], i))  # draw well pad
    layer_merged = processing.run('qgis:mergevectorlayers',
                          {'CRS': 'EPSG:3857',  # use a proper EPSG
                           'LAYERS': l1,
                           'OUTPUT': memory})['OUTPUT']
    QgsProject.instance().addMapLayer(layer_merged)
########################

######################## Combine all the detections #######################
def contain(l1, l2):  # if return True it is contained in another rect
    if l1[0] <= l2[0] and l1[1] >= l2[1] and l1[2] >= l2[2] and l1[3] <= l2[3]:
        return True, l1
    if l1[0] >= l2[0] and l1[1] <= l2[1] and l1[2] <= l2[2] and l1[3] >= l2[3]:
        return True, l2
    return False, []

def intersect(l1, l2,area_1,area_2):
    m = True
    if l1[0] >= l2[2] or l2[0] >= l1[2]:
        m = False
        return m, []
    if l1[1] <= l2[3] or l2[1] <= l1[3]:
        m = False
        return m, []
    if m == True:
        dx = min(l1[2], l2[2]) - max(l1[0], l2[0])
        dy = min(l1[1], l2[1]) - max(l1[3], l2[3])
        if (dx>=0) and (dy>=0):
            overlap_area = dx*dy
            min_area = min(area_1,area_2)
            percent_overlap = overlap_area/(min_area) 
            if percent_overlap < 0.20 :
                return False, []           
            else:    
                x_min1 = min(l1[0], l2[0])
                y_min1 = max(l1[1], l2[1])
                x_max1 = max(l1[2], l2[2])
                y_max1 = min(l1[3], l2[3])
                return m, [x_min1, y_min1, x_max1, y_max1]  # if m= True then overlap

def get_area(rect):
    w = abs(rect[2]-rect[0])
    h = abs(rect[1]-rect[3])
    area = w * h
    return area

def check_area_combine(num1,index,area,temp): 
    w = 606
    h = 606
    img_loc = get_image([num1[0],num1[1],num1[2],num1[3]],1,False,False)
    c = detect(img_loc,0.60)
    g = True
    if c!=False:
        m1=[]
#                draw_layer(x1,y1,x2,y2,'well_pad')
        m1 = find_layer(c,img_loc,[num1[0],num1[1],num1[2],num1[3]],1,w,h)
        for i in range(len(m1)):
            area_1 = get_area(m1[i])
            if abs(area_1 - area) < 90000:
                g = False
                temp[index] = m1[i]
                break
        if g == False:
            return True
        else:
            temp.pop(index)
            return False
    else:
        temp.pop(index)
        return False
        

def inter_contain(temp):
    w = 606
    h = 606
    i = 0
    length = len(temp)
    if length < 2:
        return temp
    while i <= (length - 1) and len(temp)>=2:
        num1 = temp[i]
        j = i + 1
        area_1 = get_area(num1)
#        print(len(temp),area_1,i)
#        print(num1)
        if area_1 > 250000:
            temp_1 = check_area_combine(num1,i,area_1,temp)
            if temp_1 == False:
                length = len(temp)
                continue
        
        while j <length:
            area_2 = get_area(temp[j])
#            print(len(temp),j,area_2)
#            print(j)
            if area_2 > 250000:
                temp_2 = check_area_combine(temp[j],j,area_2,temp)
                if temp_2 == False:
                    length = len(temp)
                    continue
            t2 = False
            t1 = False
            t2, l2 = intersect(temp[i], temp[j],area_1,area_2)
            if t2== True:
                temp[i] = l2
                temp.pop(j)
                i = i - 1
                length = len(temp)
                break
            t1, l1 = contain(temp[i], temp[j])
            if t1 == True:
                temp[i] = l1
                temp.pop(j)
                i = i - 1
                length = len(temp)
                break
            j = j + 1
#        if t2 == False and t1 == False:
#            i = i + 1
        i = i + 1     
    return temp
    
########################Check if image is complete or not#####
def check_complete(img_loc):
    gray = cv2.cvtColor(img_loc, cv2.COLOR_RGB2GRAY)
    no = np.sum(gray == 255)
    if no > 10000:
        return True
    else:
        return False
    
###################Search for Pads and Tanks #######################

def find_well_pad_tank(points,p2): # find the coordinates between the range of points
    start_time = time.time()
    global x_min
    global y_min
    global x_max
    global y_max
    
    x_min = points[6]
    x_max = points[2]
    y_min = points[1]
    y_max = points[5]
    poly = draw_range(points)
 #   print(x_min,x_max,y_min,y_max)
    add_1 = 100 #size of the small block of image
    add_2 = 100 # size of the large block of image
    x1 = x_min
    x2 = x1 + add_1
    img = 0 # to get image count
    row = 0
    column = 0
    print("Done Drawing")
    while x1<=x_max:
        y1 = y_min
        y2 = y1 - add_1
        row = 0
        column = column +1
        flag = 0
        while y1>=y_max:
            row = row + 1
            img = img + 1 # to get image count
            w = x2 - x1 # width of small box
            h = y1 - y2
            img_loc = []
            img_loc = get_image([x1,y1,x2,y2],img,False,False)
            if flag == 40:
                y1 = y1 - add_2
                y2 = y2 - add_2 
                flag = 0
                continue
            if check_complete(img_loc): 
                flag = flag + 1
                continue
            else:
                flag = 0
            c = detect(img_loc,0.70)
            
            if c!=False:
                print(c)
                m1=[]
#                draw_layer(x1,y1,x2,y2,'well_pad')
                m1 = find_layer(c,img_loc,[x1,y1,x2,y2],img,w,h)
                draw_under_conditions(m1,x1,x2,y1,y2)
            y1 = y1 - add_2
            y2 = y2 - add_2
            
        x1 = x1 + add_2
        x2 = x2 + add_2
if len(layers_list)!=0:
    addlayer_from_list(layers_list,"well_pad")
p = [-11601728,4996530,-11599275,4993789]
p = [-11571107,4976484,-11567279,4973184]
map =qgis.utils.iface.mapCanvas().extent()
m = [map.xMinimum(),map.yMaximum(),map.xMaximum(),map.yMinimum()]
#m=[-11569975.541365327, 4978138.834789941, -11559354.580362147, 4969962.228530273]
#m=[-11569975.541365327, 4978138.834789941,-11568295,4969990]
#m=[-11574198.908262579, 4976163.055511515, -11567149.597507296, 4968656.111598586]
#m=[-11567039.7637046426534653,4975027.5655298577621579,-11563830.2741251774132252,4971504.1490417188033462]
#m=[-11568920.9372005984187126,4979948.1961683277040720,-11559934.0365571156144142,4970082.2679200768470764]
#m=[-11569527.1735889911651611,4976197.5203683720901608, -11564460.6275777518749237,4970325.5587548026815057]
#m =[-11600530.2729765865951777,4979181.7028903672471642,-11552971.0480329040437937,4957264.8909633224830031]
#m=[-11573590.9047237243503332,4979095.3217002404853702,-11563184.0094249714165926,4969414.7527909306809306]
#m= [-11572446.3499678559601307,4981127.9333306523039937,-11560396.3935833722352982,4975574.9280383968725801]
#layer_1 = QgsProject.instance().mapLayersByName('DJStudyAreaBoundary')
#crsSrc = QgsCoordinateReferenceSystem("EPSG:26913")    # WGS 84
#crsDest = QgsCoordinateReferenceSystem("EPSG:3857")  # WGS 84 / UTM zone 33N
#transformContext = QgsProject.instance().transformContext()
#xform = QgsCoordinateTransform(crsSrc, crsDest, transformContext)
#
#ext = layer_1[0].extent()
#cord_x1 = ext.xMinimum()
#cord_x2 = ext.xMaximum() 
#cord_y1 = ext.yMaximum() 
#cord_y2 = ext.yMinimum() 
#p2 = [cord_x1,cord_y1,cord_x2,cord_y2]
p2 = m
## forward transformation: src -> dest





#feature = layer_1[0].getFeatures()
#for feat in feature:
#    print(feat.geometry())
    
#m = [519326.4096615903545171, 4505706.50590073224157095, 561449.62786065670661628, 4484401.175309794023633, 524641.80731634527910501, 4412038.34040814824402332, 483307.09421344619477168, 4434616.26304043084383011]
#pt1 = xform.transform(QgsPointXY(m[0],m[1]))
#pt2 = xform.transform(QgsPointXY(m[2],m[3]))
#pt3 = xform.transform(QgsPointXY(m[4],m[5]))
#pt4 = xform.transform(QgsPointXY(m[6],m[7]))
#print([pt1[0],pt1[1], pt2[0],pt2[1], pt3[0],pt3[1], pt4[0],pt4[1]])

#find_well_pad_tank([pt1[0],pt1[1], pt2[0],pt2[1], pt3[0],pt3[1], pt4[0],pt4[1]],p2)
find_well_pad_tank([p2[2],p2[1], p2[2],p2[3], p2[0],p2[3], p2[0],p2[1]],p2)