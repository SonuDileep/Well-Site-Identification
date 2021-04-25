map =qgis.utils.iface.mapCanvas().extent()
m = [map.xMinimum(),map.yMaximum(),map.xMaximum(),map.yMinimum()]
print("width =", abs(m[0] - m[2]))
print("height =",abs(m[3] - m[1]))