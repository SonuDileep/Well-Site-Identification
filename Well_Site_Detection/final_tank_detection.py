#Check for equipments inside well pads
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
layers_list = [] ## to store all bounding box coordinates

cfg = '/mnt/Data/YOLO_NEW/darknet-master/complete_eq/yolov4(2).cfg'
data = '/mnt/Data/YOLO_NEW/darknet-master/complete_eq/detector.data'
weights = '/mnt/Data/YOLO_NEW/darknet-master/complete_eq/yolov4_21000.weights'
thresh = 0.2
net,cn,cc = main2(cfg,data,weights)


#####################

###################### Basic Core Functions #####################

def detect_tanks(img,threshold): # for detection using YOLO
    ''' this script if you want only want get the coord '''
    res = dettect(img, net,cn,cc , thresh)
#    img = cv2.imread('/home/sonu/Desktop/ppt/-11650720.3707936374944003.302339151.png')
    
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
#            path = '/home/sonu/Desktop/ppt/' + str(y1) + str(x1) +'.png' 
#            cv2.imwrite(path,img_loc1)
            c = detect_tanks(img_loc1,0.6)
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
                    if c[i][0] != 'non':
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

#def merge_layer(new_x1, new_y1, new_x2, new_y2, i):
#    x = (new_x1 + new_x2)/2
#    y = (new_y1 + new_y2)/2
#    rect = QgsPointXY(x , y)
#    m = [rect]
#    rectangleLayer = QgsVectorLayer('Point?crs=EPSG:3857', 'well_tank',
#                                    'memory')
#    rectangleLayerProvider = rectangleLayer.dataProvider()
#    newFeat = QgsFeature()
##    symbol = QgsFillSymbol.createSimple({'name': 'square',
##                                         'color': '255,255,0,0', 'width_border': '0.8'})
##    symbol.symbolLayer(0).setStrokeColor(QColor(0, 255, 0))
##    geom = QgsGeometry.fromPolygonXY([m])
#    geom = QgsGeometry.fromWkt(rect.asWkt())
#    newFeat.setGeometry(geom)
#    rectangleLayerProvider.addFeatures([newFeat])
#     
#    return rectangleLayer

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
import json
    
with open('/home/sonu/Desktop/test/complete_pad_list.txt') as f:
    final_well_pad = json.load(f)
print(len(final_well_pad))
start_time = time.time()
well_tank_list = []
final_wellpad_list = []
print("Tank")
for i in range(len(final_well_pad)):
    if i==5:
        break
    m = check_equip(final_well_pad[i],1)
    well_tank_list.extend(m)
    if len(m)!=0:
        final_wellpad_list.append(final_well_pad[i])
print(len(final_wellpad_list))
print(len(well_tank_list))
addlayer_from_list(final_wellpad_list,"well_pad")
addlayer_from_list(well_tank_list,"well_tank")
with open('/home/sonu/Desktop/test/listfile_finalpad_3.txt', "w") as f:
    json.dump(final_wellpad_list, f)
with open('/home/sonu/Desktop/test/listfile_finaltank_3.txt', "w") as f:
    json.dump(well_tank_list, f)
tot = time.time() - start_time
print("Total_Time-",tot)