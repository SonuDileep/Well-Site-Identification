#Check for equipments inside well pads
import sys 
import time
sys.path.insert(0,'/mnt/Data/YOLO_NEW/darknet-master/')
from darknet_images_equip import main2 
from darknet_images_equip import dettect
from darknet_images_equip1 import main2 as scan2
#from darknet_images_equip1 import dettect as dettect2
import numpy as np
import math  
import json
import qimage2ndarray
import cv2
import copy
###### TO FIND THE WELL HEADS INSIDE THE RECTANGULAR AREA
from qgis.core import QgsProject
import math
import csv

layers_list = [] ## to store all bounding box coordinates
layer_from_l = True

cfg = '/mnt/Data/YOLO_NEW/darknet-master/complete_eq/yolov4(2).cfg'
data = '/mnt/Data/YOLO_NEW/darknet-master/complete_eq/detector.data'
weights = '/mnt/Data/YOLO_NEW/darknet-master/complete_eq/yolov4_21000.weights'
net,cn,cc = main2(cfg,data,weights)

cfg1 = '/mnt/Data/yolov4_ubuntu/custom_v4/yolov4(2).cfg'
data1 = '/mnt/Data/yolov4_ubuntu/custom_v4/detector.data'
weights1 = '/mnt/Data/yolov4_ubuntu/custom_v4/yolov4_6000.weights'
#net1,cn1,cc1 = scan2(cfg1,data1,weights1)
#####################

###################### Basic Core Functions #####################

def detect_tanks(img,threshold): # for detection using YOLO
    res = dettect(img, net, cn,cc, threshold)
    return res
    
#def detect(img,threshold): # for detection using YOLO
#    res1 = dettect2(img, net1, cn1,cc1, threshold)
#    return res1
    
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
    sep_list = []
    eflare_list = []
    oflare_list = []
    teqp_list = []
    whh_list = []
    pjj_list = []
    
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
#            path = '/home/sonu/Desktop/ppt/' + str(y1) + str(x1) +'.png' 
#            cv2.imwrite(path,img_loc1)
            c = detect_tanks(img_loc1,0.2)
#            print(c)
            if c!=None:
                d1 = abs(x2 - x1)
                d2 = abs(y1 - y2)
                w = d1 * 6
                h = d2 * 6
                w = 606
                h = 606
#                g = []
                for i in range(len(c)):
                    if c[i][0] == 'tank':
                        thresh = float(c[i][1])
                        if thresh > 70:
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
                            
                    if c[i][0] == 'separator':
                        thresh = float(c[i][1])
                        if thresh > 30:
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
                                sep_list.append([new_x1,new_y1,new_x2,new_y2])
                        
                    if c[i][0] == 'flare':
                        thresh = float(c[i][1])
                        if thresh > 30:
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
                                eflare_list.append([new_x1,new_y1,new_x2,new_y2])

                    if c[i][0] == 'extra_eq':
                        thresh = float(c[i][1])
                        if thresh > 20:
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
                                teqp_list.append([new_x1,new_y1,new_x2,new_y2])

  
 
                    if c[i][0] == 'pump_jack':
                        thresh = float(c[i][1])
                        if thresh > 30:
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
                                pjj_list.append([new_x1,new_y1,new_x2,new_y2])
 
                    if c[i][0] == 'well_heads':
                        thresh = float(c[i][1])
                        if thresh > 30:
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
                                whh_list.append([new_x1,new_y1,new_x2,new_y2])     
    

                    if c[i][0] == 'open_flare':
                        thresh = float(c[i][1])
                        if thresh > 30:
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
                                oflare_list.append([new_x1,new_y1,new_x2,new_y2])
#                if len(g)!=0:
#                    tank_list.extend(g)
                
            y1 = y1 - add_2
            y2 = y2 - add_2
        x1 = x1 + add_2
        x2 = x2 + add_2
    tank_list = equip_clean(tank_list)  # Combines the detections with intersections  
    tank_list = equip_clean(tank_list)
    
    sep_list = equip_clean(sep_list)  # Combines the detections with intersections  
    sep_list = equip_clean(sep_list)
    
    eflare_list = equip_clean(eflare_list)  # Combines the detections with intersections  
    eflare_list = equip_clean(eflare_list)
    
    teqp_list = equip_clean(teqp_list)  # Combines the detections with intersections  
    teqp_list = equip_clean(teqp_list)
    
    whh_list = equip_clean(whh_list)  # Combines the detections with intersections  
    whh_list = equip_clean(whh_list)
    
    oflare_list = equip_clean(oflare_list)  # Combines the detections with intersections  
    oflare_list = equip_clean(oflare_list)
    
    pjj_list = equip_clean(pjj_list)  # Combines the detections with intersections  
    pjj_list = equip_clean(pjj_list)

    
    return tank_list, sep_list, oflare_list, eflare_list, teqp_list, whh_list, pjj_list    

#########################################


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

def merge_layer1(new_x1, new_y1, i):
    x = new_x1 
    y = new_y1 
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

def merge_layer(new_x1, new_y1, new_x2, new_y2, i):
    rect = QgsRectangle(new_x1, new_y1, new_x2, new_y2)
    rectangleLayer = QgsVectorLayer('Polygon?crs=EPSG:3857', 'well_tank',
                                    'memory')
    rectangleLayerProvider = rectangleLayer.dataProvider()
    newFeat = QgsFeature()
    symbol = QgsFillSymbol.createSimple({'name': 'square',
                                         'color': '255,255,0,0', 'width_border': '0.8'})
    symbol.symbolLayer(0).setStrokeColor(QColor(0, 255, 0))
    geom = QgsGeometry.fromWkt(rect.asWktPolygon())
    newFeat.setGeometry(geom)
    rectangleLayerProvider.addFeatures([newFeat])
    return rectangleLayer

def addlayer_from_list(list_name,name):
    memory = 'memory:' + name
    l1 = []
    if layer_from_l==True:
        for i in range(len(list_name)):
            l1.append(merge_layer(list_name[i][0], list_name[i][1],list_name[i][2],list_name[i][3], i))  # draw well pad
    if layer_from_l==False:
        for i in range(len(list_name)):
            l1.append(merge_layer1(list_name[i][0], list_name[i][1], i))  # draw well pad
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
        
def return_center(full_equip):
    a1 = 0
    a2 = 0
    for j in range(len(full_equip)):
        if full_equip[j]!=None:
            x1 = 0
            x2 = 0
            x1 = (full_equip[j][1] + full_equip[j][3])/2
            x2 = (full_equip[j][0] + full_equip[j][2])/2   
            if j!=0: 
                a1 = (a1 + x1)/2
                a2 = (a2 + x2)/2
            if j==0: 
                a1 = x1
                a2 = x2
    return [a2,a1]
    

temp_loc = '/mnt/Data/Final_Code_DJ/Final_txt_list' +"/"+ "Data_well_pad.csv"
with open(temp_loc, 'w+') as csvfile: #creating a csv file
    filewriter =  csv.writer(csvfile, delimiter=',', escapechar=' ', quoting=csv.QUOTE_NONE) 
    filewriter.writerow(['Name', 'Latitude', 'Longitude', 'Tank Count', 'Separators', 'Enclosed Flare','Open Flare','Well Heads','Pump Jack','Total Equipment' ])
#    filewriter.writerow("\n")    
###################Search for Pads and Tanks #######################
with open('/home/sonu/Desktop/test/complete_pad_list.txt') as f:
    final_well_pad = json.load(f)
print(len(final_well_pad))
start_time = time.time()
my_well_pad = []

well_tank_list = []
separator_list = []
enclosed_flare_list = []
open_flare_list = []
other_equip_list = []
well_head_list = []
pj_list = []

final_wellpad_list = []
each_equip_at_loc = []
for i in range(len(final_well_pad)):
#    draw_layer(final_well_pad[i][0],final_well_pad[i][1],final_well_pad[i][2],final_well_pad[i][3],'well_pad')        
    full_equip = []
    
    tank = []
    separator = []
    eflare = []
    oflare = []
    oeq = []
    wh = []
    pj = []
    
    tank,separator,eflare, oflare, oeq, wh, pj = check_equip(final_well_pad[i],1)
#    width1 = abs(final_well_pad[i][2] - final_well_pad[i][0])
#    height1 = abs(final_well_pad[i][1] - final_well_pad[i][3])
#    if len(tank)==0 and len(separator)>0:
#        if width1>150 or height1>150:
#            continue
#        if 90<width1<150 or 90<height1<150:
#            img_loc = []
#            w_diff = abs(width1 - height1)
#            if width1>height1:
#                x1111 = final_well_pad[i][0] - 20
#                x2222 = final_well_pad[i][2] + 20
#                y1111 = final_well_pad[i][1] + w_diff
#                y2222 = final_well_pad[i][3] - w_diff
#            if width1<=height1:
#                x1111 = final_well_pad[i][0] - w_diff
#                x2222 = final_well_pad[i][2] + w_diff
#                y1111 = final_well_pad[i][1] + 20
#                y2222 = final_well_pad[i][3] - 20
#            img_loc = get_image([x1111,y1111,x2222,y2222],1,False,False)                        
#            c = detect(img_loc,0.50)
#            if c==None:
#                continue   
#                
#    if 0<len(tank)<=2 and len(separator)==0:
#        if width1>400 or height1>400:
#            continue
#        if 200<width1<400 or 200<height1<400:
#            img_loc = []
#            w_diff = abs(width1 - height1)
#            if width1>height1:
#                x1111 = final_well_pad[i][0] - 20
#                x2222 = final_well_pad[i][2] + 20
#                y1111 = final_well_pad[i][1] + w_diff
#                y2222 = final_well_pad[i][3] - w_diff
#            if width1<=height1:
#                x1111 = final_well_pad[i][0] - w_diff
#                x2222 = final_well_pad[i][2] + w_diff
#                y1111 = final_well_pad[i][1] + 20
#                y2222 = final_well_pad[i][3] - 20
#            img_loc = get_image([x1111,y1111,x2222,y2222],1,False,False)                        
#            c = detect(img_loc,0.50)
#            if c== None:
#                continue                           
    m1 = 0
    m2 = 0
    m3 = 0
    if len(tank)!=0:
        m1 = len(tank[0])
    if len(separator)!=0:
        m2 = len(separator[0])
    if len(pj)!=0:
        m3 = len(pj[0])
        
    if m1!=0 or m2!=0 or m3!=0:
        if len(tank)!=0:
            well_tank_list.extend(tank)
        if len(separator)!=0:
            separator_list.extend(separator)
        if len(pj)!=0:
            pj_list.extend(pj)            
        if len(wh)!=0:
            well_head_list.extend(wh)
        enclosed_flare_list.extend(eflare)  
        open_flare_list.extend(oflare) 
        if len(oeq)!=0:
            other_equip_list.extend(oeq)
#        print(final_well_pad[i])    
        final_wellpad_list.append(final_well_pad[i])
        
        a1=[]
        a2=[]
        a3=[]
        a4=[]
        a5=[]
        a6=[]
        a7=[]
        
        a1 = copy.deepcopy(tank)
        a2 = copy.deepcopy(separator)
        a3 = copy.deepcopy(eflare)
        a4 = copy.deepcopy(oflare)
        a5 = copy.deepcopy(oeq)
        a6 = copy.deepcopy(wh)
        a7 = copy.deepcopy(pj)
        
        if len(tank)!=0:
            full_equip.extend(a1)
        if len(separator)!=0:
            full_equip.extend(a2)
        if len(eflare)!=0:
            full_equip.extend(a3)
        if len(oflare)!=0:
            full_equip.extend(a4)
        if len(oeq)!=0:
            full_equip.extend(a5)
#        if len(wh)!=0:
#            full_equip.extend(a6)
        if len(pj)!=0:
            full_equip.extend(a7)
            
        pads_new = []
        temppp = []
        temppp = copy.deepcopy(full_equip)
        pads_new = return_center(full_equip)
        each_equip_at_loc.append(temppp)
        my_well_pad.append(pads_new)
#        print(len(full_equip))
        x = pads_new[0]
        y = pads_new[1]
        lon1 = x *  180 / 20037508.34 ;
        lat1 = math.atan(math.exp(y * math.pi / 20037508.34)) * 360 / math.pi - 90; 
        temp_loc = '/mnt/Data/Final_Code_DJ/Final_txt_list' +"/"+ "Data_well_pad.csv"
        if len(tank)!=0 or len(separator)!=0:
            with open(temp_loc, 'a+') as csvfile: #creating a csv file  
                filewriter =  csv.writer(csvfile, delimiter=',', escapechar=' ', quoting=csv.QUOTE_NONE) 
                filewriter.writerow(["Well_Pad", lat1, lon1, len(tank), len(separator), len(oflare),len(eflare),len(wh),len(pj),len(full_equip)]) 
       
        else:
            if len(pj)!=0 :
                with open(temp_loc, 'a+') as csvfile: #creating a csv file  
                    filewriter =  csv.writer(csvfile, delimiter=',', escapechar=' ', quoting=csv.QUOTE_NONE)
                    filewriter.writerow(["Pump_Jack", lat1, lon1, len(tank), len(separator), len(oflare),len(eflare),len(wh),len(pj),len(full_equip)])        
        
        
#print("Well Pad Count - ",len(final_wellpad_list))
#print("Well Tank Count - ",len(well_tank_list))
#print("Separator Count - ",len(separator_list))
#print("Flare Count - ",len(enclosed_flare_list))
#print("Separator Count - ",len(open_flare_list))
#print("Flare Count - ",len(pj_list))
#print("Flare Count - ",len(well_head_list))
if len(final_wellpad_list)!=0:
    addlayer_from_list(final_wellpad_list,"well_pad")
if len(well_tank_list)!=0:
    addlayer_from_list(well_tank_list,"well_tank")
if len(separator_list)!=0:
    addlayer_from_list(separator_list,"separator")
if len(enclosed_flare_list)!=0:
    addlayer_from_list(enclosed_flare_list,"open_flare")
if len(open_flare_list)!=0:
    addlayer_from_list(open_flare_list,"enclosed_flare")
if len(pj_list)!=0:
    addlayer_from_list(pj_list,"pump_jack")
if len(well_head_list)!=0:
    addlayer_from_list(well_head_list,"well_head")
layer_from_l = False
if len(my_well_pad)!=0:
    addlayer_from_list(my_well_pad,"final_pad")
    
with open('/home/sonu/Desktop/test/finalpad_3.txt', "w") as f:
    json.dump(final_wellpad_list, f)
    
with open('/home/sonu/Desktop/test/finaltank_3.txt', "w") as f:
    json.dump(well_tank_list, f)
with open('/home/sonu/Desktop/test/finalseparator_3.txt', "w") as f:
    json.dump(separator_list, f)
with open('/home/sonu/Desktop/test/finalenclflare_3.txt', "w") as f:
    json.dump(open_flare_list, f) 
with open('/home/sonu/Desktop/test/finalopenflare_3.txt', "w") as f:
    json.dump(enclosed_flare_list, f)
with open('/home/sonu/Desktop/test/finalpj_3.txt', "w") as f:
    json.dump(pj_list, f)
with open('/home/sonu/Desktop/test/finalwh_3.txt', "w") as f:
    json.dump(well_head_list, f) 
    
with open('/home/sonu/Desktop/test/pad_listt.txt', "w") as f: # 2D points
    json.dump(my_well_pad, f)
with open('/home/sonu/Desktop/test/each_equip_at_loc.txt', "w") as f:
    json.dump(each_equip_at_loc, f)
tot = time.time() - start_time
print("Total_Time-",tot)