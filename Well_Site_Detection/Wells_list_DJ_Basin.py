import numpy as np
import pandas as pd
import math
import numpy as np
from numpy.linalg import norm
import time
import json

#Euclidean Distance formulae
def dist(p1,p2):
    distance = math.sqrt(((p1[0]-p2[0])**2)+((p1[1]-p2[1])**2))
    return distance
#Finds the perpendicular distance from one point to line    
def short_dist(p1,p2,p3): #perpendicular distance
    p1=np.array(p1)
    p2=np.array(p2)
    p3=np.array(p3)
    d=np.cross(p2-p1,p3-p1)/np.linalg.norm(p2-p1) 
    return d
    
#Return the closest distance from point to rectangle    
def find_distance(rect,point):
    p1 = [rect[0],rect[3]]
    p2 = [rect[2],rect[3]] 
    p3 = [rect[2],rect[1]]
    p4 = [rect[0],rect[1]]
    if point[1]>=rect[1]:
        if point[0]<rect[0]: 
            return dist(point, p4)
        if point[0]>rect[2]:
            return dist(point,p3)
        else:
            return short_dist(p3,p4,point)
    if point[1]<=rect[3]:
        if point[0]<rect[0]: 
            return dist(point, p1)
        if point[0]>rect[2]:
            return dist(point,p2) 
        else:
            return short_dist(p1,p2,point)
    elif point[0]>=rect[2]:
            return short_dist(p2,p3,point)  
    elif point[0]<=rect[0]:
            return short_dist(p1,p4,point)


def if_in_map_canvas(x1, y1,cord):
    if x1 >= cord[0] and y1 <= cord[1] and x1 <= cord[4] and y1 >= cord[5]:
        return True
    else:
        return False
    
def draw_range(cord): # draws the rectangular search space ( 4 input points from user)
    layer = QgsVectorLayer('Polygon?crs=EPSG:3857', 'area', 'memory')
    pr = layer.dataProvider()
    poly = QgsFeature()
    points = [QgsPointXY(cord[0], cord[1]),
              QgsPointXY(cord[2], cord[3]),
              QgsPointXY(cord[4], cord[5]),
              QgsPointXY(cord[6], cord[7])]
    poly.setGeometry(QgsGeometry.fromPolygonXY([points]))
    return poly 

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
    
def addlayer_from_list(list_name,name):
    memory = 'memory:' + name
    l1 = []
    layer_from_l = False
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

crsSrc = QgsCoordinateReferenceSystem("EPSG:26913")    # WGS 84
crsDest = QgsCoordinateReferenceSystem("EPSG:3857")  # WGS 84 / UTM zone 33N
transformContext = QgsProject.instance().transformContext()
xform = QgsCoordinateTransform(crsSrc, crsDest, transformContext)

m = [519326.4096615903545171, 4505706.50590073224157095, 561449.62786065670661628, 4484401.175309794023633, 524641.80731634527910501, 4412038.34040814824402332, 483307.09421344619477168, 4434616.26304043084383011]
pt1 = xform.transform(QgsPointXY(m[0],m[1]))
pt2 = xform.transform(QgsPointXY(m[2],m[3]))
pt3 = xform.transform(QgsPointXY(m[4],m[5]))
pt4 = xform.transform(QgsPointXY(m[6],m[7]))

poly = draw_range([pt1[0],pt1[1], pt2[0],pt2[1], pt3[0],pt3[1], pt4[0],pt4[1]]) 

crsSrc = QgsCoordinateReferenceSystem("EPSG:26913")    # WGS 84
crsDest = QgsCoordinateReferenceSystem("EPSG:3857")  # WGS 84 / UTM zone 33N

transformContext = QgsProject.instance().transformContext()
xform = QgsCoordinateTransform(crsSrc, crsDest, transformContext)

layer = QgsProject.instance().mapLayersByName('Wells')
layer_1 = QgsProject.instance().mapLayersByName('DJ_Basin')
ext = layer_1[0].extent()
cord_x1 = ext.xMinimum()
cord_x2 = ext.xMaximum() 
cord_y1 = ext.yMaximum() 
cord_y2 = ext.yMinimum() 
cord = [cord_x1,cord_y1,cord_x2,cord_y1,cord_x2,cord_y2,cord_x1,cord_y2]
#print(cord)
i = 0
a = []    
    
for f in layer[0].getFeatures():
    geom = f.geometry()
    p = xform.transform(QgsPointXY(geom.asPoint().x(), geom.asPoint().y()))
    if poly.geometry().contains(QgsPointXY(p)) == True:
        a.append([p[0],p[1]])

data = pd.read_csv("/mnt/Data/Final_Code_DJ/Final_well_pad_list/well_locations.csv") 
wells_filter = []

    
crsSrc = QgsCoordinateReferenceSystem("EPSG:4326")    # WGS 84
crsDest = QgsCoordinateReferenceSystem("EPSG:3857")  # WGS 84 / UTM zone 33N
transformContext = QgsProject.instance().transformContext()
xform = QgsCoordinateTransform(crsSrc, crsDest, transformContext)

with open('/mnt/Data/Final_Code_DJ/Final_well_pad_list/wells_linked.txt') as f:
    layers_list1 = json.load(f)

p = 0
final_wells_list = [] 
final_pad_well = []# Filtered List of all well pads and wells producing
print(len(data))
for index, row in data.iterrows():
#    p = p + 1
#    if p==1000:
#        break
#   wells_filter.append([row[31],row[32]])
#    if row[17]=='PR' or row[17]=='IJ':
    pt1 = xform.transform(QgsPointXY(row[32],row[31]))
    for i in range(len(a)):
            if abs(a[i][0] - pt1[0])<=2 and abs(a[i][1] - pt1[1])<=2:
##                    print(layers_list1[i][k][0], pt1[0])
#                final_wells_list.append([a[i][0],a[i][1]])
                if len(str(row[1]))==1:
                    county = str('00') + str(row[1])
                if len(str(row[1]))==2:
                    county = str('0') + str(row[1])
                if len(str(row[1]))==3:
                    county = str(row[1])


                if len(str(row[2]))==4:
                    api_well = str('0') + str(row[2])
                if len(str(row[2]))==3:
                    api_well = str('00') + str(row[2])
                if len(str(row[2]))==5:
                    api_well = str(row[2])
                    
                api_seq = str('05') + county + api_well
                final_wells_list.append(api_seq)

temp_loc = '/mnt/Data/Final_Code_DJ/DJ_Basin_Wells_1.csv'
import csv
#for y in range(len(final_wells_list)):
with open(temp_loc, 'a',newline='') as csvfile: #creating a csv file  
#    filewriter =  csv.writer(csvfile, delimiter=',', escapechar=' ', quoting=csv.QUOTE_NONE) 
    filewriter =  csv.writer(csvfile)
    for y in range(len(final_wells_list)):
        filewriter.writerows([final_wells_list[y]])  
         
#There are 10634 wells with PR and IJ Status






