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
weights = '/mnt/Data/YOLO_NEW/darknet-master/complete_eq/yolov4_last.weights'
thresh = 0.5
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

def draw_range(cord): # draws the rectangular search space ( 4 input points from user)
    layer = QgsVectorLayer('Polygon?crs=EPSG:3857', 'area', 'memory')
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

#########################################


##################################################################
            
################## QGIS Layer Implementation #######################        


def merge_layer(new_x1, new_y1, i):
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

def addlayer_from_list(list_name,name):
    memory = 'memory:' + name
    l1 = []
    for i in range(len(list_name)):
        l1.append(merge_layer(list_name[i][0], list_name[i][1], i))  # draw well pad
    layer_merged = processing.run('qgis:mergevectorlayers',
                          {'CRS': 'EPSG:3857',  # use a proper EPSG
                           'LAYERS': l1,
                           'OUTPUT': memory})['OUTPUT']
    QgsProject.instance().addMapLayer(layer_merged)
########################

    
########################Check if image is complete or not#####
def check_complete(img_loc):
    gray = cv2.cvtColor(img_loc, cv2.COLOR_RGB2GRAY)
    no = np.sum(gray == 255)
    if no > 10000:
        return True
    else:
        return False
        
def calc_dist(x1, y1, x2, y2):
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)         
###################Search for Well Heads #######################
layer = QgsProject.instance().mapLayersByName('Wells')
m = [519326.4096615903545171, 4505706.50590073224157095, 561449.62786065670661628, 4484401.175309794023633, 524641.80731634527910501, 4412038.34040814824402332, 483307.09421344619477168, 4434616.26304043084383011]
crsSrc = QgsCoordinateReferenceSystem("EPSG:26913")    # WGS 84
crsDest = QgsCoordinateReferenceSystem("EPSG:3857")  # WGS 84 / UTM zone 33N
transformContext = QgsProject.instance().transformContext()
xform = QgsCoordinateTransform(crsSrc, crsDest, transformContext)
pt1 = xform.transform(QgsPointXY(m[0],m[1]))
pt2 = xform.transform(QgsPointXY(m[2],m[3]))
pt3 = xform.transform(QgsPointXY(m[4],m[5]))
pt4 = xform.transform(QgsPointXY(m[6],m[7]))
p = [pt1[0],pt1[1], pt2[0],pt2[1], pt3[0],pt3[1], pt4[0],pt4[1]]
print(p)
#map =qgis.utils.iface.mapCanvas().extent()
#m = [map.xMinimum(),map.yMaximum(),map.xMaximum(),map.yMinimum()]

#p = [m[0],m[1],m[0],m[3],m[2],m[3],m[2],m[1]]
#print(p)
poly = draw_range(p)
a = []
for f in layer[0].getFeatures():
    geom = f.geometry()
    if poly.geometry().contains(QgsPointXY(xform.transform(QgsPointXY(geom.asPoint().x(), geom.asPoint().y())))) == True:
        pt1 = xform.transform(QgsPointXY(geom.asPoint().x(), geom.asPoint().y()))
        a.append(pt1)
print(len(a))
print(a[1])
sizes = [30,40]
well_head_comb = []
well_head = []
pump_jack = []
skip_loc = []
for k in range(len(a)):      
    g = []
    g1 = []
    g2 = []
    tempo = 0
    for l in range(len(sizes)):
#        pt1 = xform.transform(QgsPointXY(a[k][0],a[k][1]))
#        print(pt1)
        pt1 = a[k]
#        print(pt1)
        x11 =  pt1[0]- sizes[l]
        y11 = pt1[1] + sizes[l]
        x22 = pt1[0] + sizes[l]
        y22 = pt1[1] - sizes[l]
        img_loc = get_image([x11, y11, x22, y22], 0, False,False)
        c = detect_tanks(img_loc,0.2)

        if check_complete(img_loc):
            skip_loc.append(a[k])
            
        else:
            tempo = 0
            if c!=False:
                m1=[]
                x1 = x11
                y1 = y11
                d1 = sizes[l] * 2
                d2 = sizes[l] * 2
                #d1 = cord.xMaximum() - cord.xMinimum()
                #d2 = cord.yMaximum() - cord.yMinimum()
                w = 606
                h = 606
                for i in range(len(c)):
                    if c[i][0] =='well_heads' :
                        x11, y11, w_size, h_size = c[i][2]
                        x_start = round(x11 - (w_size/2))
                        y_start = round(y11 - (h_size/2))
                        x_end = x_start + w_size
                        y_end = y_start + h_size
                        new_x1 = x1 + (d1/w) * x_start
                        new_x2 = x1 + (d1/w) * x_end
                        new_y1 = y1 - (d2/h) * y_start
                        new_y2 = y1 - (d2/h) * y_end
                        mid_x = (new_x1 + new_x2)/2
                        mid_y = (new_y1 + new_y2)/2
                        distance = calc_dist(mid_x,mid_y,pt1[0],pt1[1])
#                        print(distance)
                        if distance < 25:
                            if (pt1[0],pt1[1]) not in g:
                                g.append((pt1[0],pt1[1]))
                                g1.append((pt1[0],pt1[1]))

                    if c[i][0] =='pump_jack' :
                        x11, y11, w_size, h_size = c[i][2]
                        x_start = round(x11 - (w_size/2))
                        y_start = round(y11 - (h_size/2))
                        x_end = x_start + w_size
                        y_end = y_start + h_size
                        new_x1 = x1 + (d1/w) * x_start
                        new_x2 = x1 + (d1/w) * x_end
                        new_y1 = y1 - (d2/h) * y_start
                        new_y2 = y1 - (d2/h) * y_end
                        mid_x = (new_x1 + new_x2)/2
                        mid_y = (new_y1 + new_y2)/2
                        distance = calc_dist(mid_x,mid_y,pt1[0],pt1[1])
#                        print(distance)
                        if distance < 25:
                            if (pt1[0],pt1[1]) not in g:
                                g.append((pt1[0],pt1[1]))    
                                g2.append((pt1[0],pt1[1]))
#                        draw_layer(new_x1,new_y1,new_x2,new_y2,'det')
    well_head_comb.extend(g)   
    well_head.extend(g1) 
    pump_jack.extend(g2) 
with open('/mnt/Data/Final_Code_DJ/wellhead/well_head_comb.txt', "w") as f:
    json.dump(well_head_comb, f)     
with open('/mnt/Data/Final_Code_DJ/wellhead/well_head.txt', "w") as f:
    json.dump(well_head, f) 
with open('/mnt/Data/Final_Code_DJ/wellhead/pump_jack.txt', "w") as f:
    json.dump(pump_jack, f) 
if len(well_head_comb)!=0:
    addlayer_from_list(well_head_comb,"well_head_comb")
if len(well_head)!=0:
    addlayer_from_list(well_head,"well_head")
if len(pump_jack)!=0:
    addlayer_from_list(pump_jack,"pump_jack")
with open('/mnt/Data/Final_Code_DJ/wellhead/skip_loc.txt', "w") as f:
    json.dump(skip_loc, f) 
    
#                        loc = "/mnt/Data/Final_Code_DJ/img/w1"+str(k)+".jpg"
#                        cv2.imwrite(loc,img_loc) 
