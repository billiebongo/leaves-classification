
import cv2
import numpy as np
import pandas as pd
from os import listdir
from os.path import isfile, join
import os


OUTPUT_PATH = 'test5'



#Given drawn rectangle, find area and ratio
def find_height_width(img_no,h,w): # vert/hori/square and ratio
	area= h*w
	ratio=h/w
	return area, ratio

# find centroids of the leaf in image
def find_moments(cnt):
	moments = cv2.moments(cnt)
	cx = int(moments['m10']/moments['m00']) # cx = M10/M00
	cy = int(moments['m01']/moments['m00']) # cy = M01/M00
	return (cx,cy)

#draw box tightly enclosing the leave in the image
def find_rectangle(img,cnt):
	x,y,w,h = cv2.boundingRect(cnt)
	cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
	print("width: {}".format(w))
	print("height: {}".format(h))
	return h,w

#Use algo to draw hull of leaves and return large pit and small pit counts by convexity defect lengths
def evalLeaf(img_no):
	# first one is source image, second is contour retrieval mode, third is contour approximation method. 
	im = cv2.imread('{}'.format(img_no))
	imgray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
	ret,thresh = cv2.threshold(imgray,127,255,0)
	image, contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
	print(contours)
	img1 = im.copy()
	max_area=0
	for i in range(len(contours)):
			cnt=contours[i]
			area = cv2.contourArea(cnt)
			if(area>max_area):
				max_area=area
				ci=i
	cnt=contours[ci]


	img = cv2.drawContours(img1, [cnt], 0, (0,255,0), 3) 
	#hull = [cv2.convexHull(c) for c in contours]
	hull = cv2.convexHull(cnt,returnPoints = False) #draw convexity hull
	drawing = np.zeros(img.shape,np.uint8)
	cv2.drawContours(drawing,[cnt],0,(0,255,0),2)
	#imghull=cv2.drawContours(drawing,[hull],0,(0,0,255),2)
	#cv2.imshow('input',imghull)
	#cv2.waitKey(0) #to pause and display image analysis output
	defects = cv2.convexityDefects(cnt, hull)
	# [ start point, end point, farthest point, approx distance to farthest point ]
	print(defects)
	#cv2.imshow('img1.jpg',img1)
	small_pit=0
	large_pit=0
	count=0
	defects_count=0
	for x in range(defects.shape[0]):
		s, e, f, d = defects[x, 0]
		defects_count+=1
		if 6000 <= d:
			far = tuple(cnt[f][0])
			if count==0:
				cv2.circle(img1, far, 40, [0, 255, 0], 5)
				count+=1
			else:
				if 24000 <= d: #if convexity defect >24 000
					cv2.circle(img1, far, 40, [0, 255, 0], 5) #draw a circle at green large pit
					large_pit+=1
				else: #if convexity defect between 6000 and 24 000
					cv2.circle(img1, far, 40, [0, 255, 255], 5) #draw a circle at yellow small pit
					small_pit+=1
			print("dound")
			start = tuple(cnt[s][0])
			end = tuple(cnt[e][0])
			cv2.line(img1, start, end, [0, 255, 0], 2)
		else:
			continue
	print("output_path")
	print('{}/{}'.format(OUTPUT_PATH,img_no))
	cv2.imwrite('{}/{}'.format(OUTPUT_PATH,img_no),img1)
	moments_x, moments_y=find_moments(cnt)
	h,w=find_rectangle(img1,cnt)
	area, ratio=find_height_width(img_no, h,w)
	cv2.destroyAllWindows()
	return moments_x, moments_y, area, ratio, small_pit, large_pit, defects_count

def runEval(list_of_imgs):
	df = pd.DataFrame()
	for i in range(len(list_of_imgs)):
		moments_x,moments_y, area, ratio, small_pit, large_pit, defects_count=evalLeaf(list_of_imgs[i])
		df=df.append({'id':list_of_imgs[i],'moments_x':moments_x,'moments_y': moments_y, 'wxh':area, 'ratio':ratio, 's_pit':small_pit, 'l_pit': large_pit, 'def_count': defects_count}, ignore_index=True)

	return df

#test_imgs=['1139.jpg', '53.jpg', '659.jpg', '1028.jpg', '1542.jpg', '1338.jpg', '406.jpg', '577.jpg', '1150.jpg', '845.jpg', '820.jpg', '1233.jpg', '1143.jpg', '1227.jpg', '65.jpg', '135.jpg', '674.jpg', '264.jpg', '1331.jpg', '1214.jpg', '974.jpg', '1047.jpg', '783.jpg', '835.jpg', '359.jpg', '504.jpg', '232.jpg', '1295.jpg', '1084.jpg', '655.jpg', '1033.jpg', '1158.jpg', '385.jpg', '138.jpg', '300.jpg', '46.jpg', '818.jpg', '1060.jpg', '1485.jpg', '1332.jpg', '947.jpg', '1259.jpg', '287.jpg', '12.jpg', '1069.jpg', '1114.jpg', '965.jpg', '648.jpg', '1385.jpg', '627.jpg', '1484.jpg', '884.jpg', '255.jpg', '1053.jpg', '707.jpg', '16.jpg', '250.jpg', '77.jpg', '584.jpg', '1565.jpg', '1109.jpg', '1038.jpg', '113.jpg', '1188.jpg', '1237.jpg', '1540.jpg', '1363.jpg', '1333.jpg', '1103.jpg', '500.jpg', '1092.jpg', '1415.jpg', '271.jpg', '1078.jpg', '177.jpg', '964.jpg', '1490.jpg', '372.jpg', '857.jpg', '715.jpg', '200.jpg', '795.jpg', '1177.jpg', '1369.jpg', '1290.jpg', '814.jpg', '1583.jpg', '1573.jpg', '708.jpg', '195.jpg', '68.jpg', '639.jpg', '308.jpg', '1111.jpg', '401.jpg', '922.jpg', '578.jpg', '91.jpg', '833.jpg', '234.jpg', '1580.jpg', '1318.jpg', '1248.jpg', '7.jpg', '1241.jpg', '1185.jpg', '402.jpg', '193.jpg', '686.jpg', '229.jpg', '1234.jpg', '507.jpg', '690.jpg', '533.jpg', '343.jpg', '1215.jpg', '546.jpg', '902.jpg', '666.jpg', '734.jpg', '1510.jpg', '683.jpg', '1212.jpg', '1293.jpg', '1567.jpg', '761.jpg', '121.jpg', '1462.jpg', '1224.jpg', '626.jpg', '1029.jpg', '791.jpg', '1079.jpg', '746.jpg', '141.jpg', '274.jpg', '272.jpg', '1199.jpg', '1107.jpg', '1430.jpg', '903.jpg', '1102.jpg', '154.jpg', '261.jpg', '472.jpg', '1498.jpg', '197.jpg', '872.jpg', '834.jpg', '1466.jpg', '208.jpg', '320.jpg', '739.jpg', '62.jpg', '156.jpg', '1376.jpg', '670.jpg', '1050.jpg', '1453.jpg', '486.jpg', '332.jpg', '432.jpg', '1068.jpg', '28.jpg', '1064.jpg', '590.jpg', '57.jpg', '1267.jpg', '473.jpg', '351.jpg', '1140.jpg', '59.jpg', '1246.jpg', '1274.jpg', '1247.jpg', '1304.jpg', '887.jpg', '1138.jpg', '604.jpg', '1351.jpg', '560.jpg', '361.jpg', '1133.jpg', '254.jpg', '775.jpg', '1058.jpg', '725.jpg', '94.jpg', '9.jpg', '474.jpg', '301.jpg', '613.jpg', '691.jpg', '1558.jpg', '123.jpg', '967.jpg', '1057.jpg', '119.jpg', '104.jpg', '1105.jpg', '221.jpg', '453.jpg', '1535.jpg', '1008.jpg', '544.jpg', '1458.jpg', '292.jpg', '483.jpg', '70.jpg', '1361.jpg', '448.jpg', '476.jpg', '1553.jpg', '1091.jpg', '74.jpg', '1307.jpg', '1280.jpg', '1149.jpg', '36.jpg', '607.jpg', '223.jpg', '846.jpg', '744.jpg', '207.jpg', '701.jpg', '688.jpg', '1533.jpg', '946.jpg', '941.jpg', '97.jpg', '957.jpg', '1203.jpg', '170.jpg', '503.jpg', '89.jpg', '1264.jpg', '557.jpg', '1412.jpg', '157.jpg', '1279.jpg', '829.jpg', '527.jpg', '384.jpg', '23.jpg', '174.jpg', '1009.jpg', '158.jpg', '945.jpg', '1055.jpg', '650.jpg', '608.jpg', '1122.jpg', '368.jpg', '743.jpg', '480.jpg', '1193.jpg', '299.jpg', '184.jpg', '162.jpg', '982.jpg', '653.jpg', '518.jpg', '1127.jpg', '1422.jpg', '953.jpg', '509.jpg', '1487.jpg', '137.jpg', '1464.jpg', '781.jpg', '1314.jpg', '925.jpg', '1406.jpg', '1537.jpg', '1357.jpg', '751.jpg', '909.jpg', '998.jpg', '1288.jpg', '210.jpg', '391.jpg', '4.jpg', '1137.jpg', '836.jpg', '447.jpg', '1018.jpg', '105.jpg', '980.jpg', '244.jpg', '95.jpg', '1400.jpg', '1315.jpg', '736.jpg', '1261.jpg', '624.jpg', '1181.jpg', '699.jpg', '1192.jpg', '222.jpg', '832.jpg', '405.jpg', '1116.jpg', '1418.jpg', '799.jpg', '913.jpg', '719.jpg', '1209.jpg', '150.jpg', '905.jpg', '579.jpg', '192.jpg', '583.jpg', '921.jpg', '888.jpg', '1354.jpg', '1517.jpg', '1526.jpg', '41.jpg', '515.jpg', '1481.jpg', '434.jpg', '512.jpg', '549.jpg', '625.jpg', '1522.jpg', '830.jpg', '1297.jpg', '1486.jpg', '370.jpg', '1416.jpg', '457.jpg', '1303.jpg', '182.jpg', '151.jpg', '679.jpg', '1191.jpg', '373.jpg', '1470.jpg', '1404.jpg', '347.jpg', '484.jpg', '441.jpg', '136.jpg', '1108.jpg', '565.jpg', '891.jpg', '1560.jpg', '114.jpg', '147.jpg', '1252.jpg', '220.jpg', '973.jpg', '144.jpg', '191.jpg', '536.jpg', '1439.jpg', '1161.jpg', '1451.jpg', '111.jpg', '935.jpg', '295.jpg', '1183.jpg', '1022.jpg', '580.jpg', '185.jpg', '567.jpg', '1146.jpg', '1260.jpg', '1489.jpg', '1447.jpg', '554.jpg', '1189.jpg', '1426.jpg', '13.jpg', '531.jpg', '1371.jpg', '39.jpg', '817.jpg', '86.jpg', '279.jpg', '643.jpg', '1528.jpg', '658.jpg', '790.jpg', '128.jpg', '131.jpg', '421.jpg', '176.jpg', '1074.jpg', '98.jpg', '526.jpg', '395.jpg', '1493.jpg', '1579.jpg', '1368.jpg', '1362.jpg', '429.jpg', '1129.jpg', '1141.jpg', '705.jpg', '750.jpg', '1020.jpg', '1409.jpg', '638.jpg', '99.jpg', '230.jpg', '843.jpg', '895.jpg', '1272.jpg', '1456.jpg', '180.jpg', '1383.jpg', '1343.jpg', '1063.jpg', '1026.jpg', '1513.jpg', '1162.jpg', '1336.jpg', '1151.jpg', '1070.jpg', '316.jpg', '943.jpg', '352.jpg', '1045.jpg', '1427.jpg', '52.jpg', '1313.jpg', '305.jpg', '1465.jpg', '977.jpg', '890.jpg', '541.jpg', '1455.jpg', '889.jpg', '284.jpg', '1401.jpg', '1190.jpg', '880.jpg', '125.jpg', '1164.jpg', '403.jpg', '51.jpg', '159.jpg', '285.jpg', '1397.jpg', '611.jpg', '696.jpg', '731.jpg', '1577.jpg', '424.jpg', '276.jpg', '780.jpg', '723.jpg', '281.jpg', '1429.jpg', '1155.jpg', '1204.jpg', '381.jpg', '47.jpg', '1298.jpg', '640.jpg', '1478.jpg', '842.jpg', '479.jpg', '1178.jpg', '1195.jpg', '991.jpg', '1106.jpg', '617.jpg', '575.jpg', '534.jpg', '297.jpg', '1445.jpg', '471.jpg', '1012.jpg', '1564.jpg', '687.jpg', '703.jpg', '819.jpg', '1364.jpg', '1546.jpg', '1497.jpg', '251.jpg', '112.jpg', '804.jpg', '997.jpg', '337.jpg', '1082.jpg', '102.jpg', '442.jpg', '929.jpg', '414.jpg', '1071.jpg', '574.jpg', '477.jpg', '465.jpg', '1044.jpg', '346.jpg', '44.jpg', '1389.jpg', '644.jpg', '1306.jpg', '1437.jpg', '407.jpg', '1043.jpg', '161.jpg', '786.jpg', '1015.jpg', '1382.jpg', '599.jpg', '495.jpg', '612.jpg', '296.jpg', '266.jpg', '399.jpg', '1316.jpg', '770.jpg', '318.jpg', '555.jpg', '1285.jpg', '1334.jpg', '277.jpg', '668.jpg', '313.jpg', '126.jpg', '33.jpg', '110.jpg', '1054.jpg', '187.jpg', '1576.jpg', '400.jpg', '988.jpg', '19.jpg', '1207.jpg', '79.jpg', '378.jpg', '1067.jpg', '930.jpg', '439.jpg', '90.jpg', '764.jpg', '1035.jpg', '24.jpg', '1428.jpg', '1434.jpg', '293.jpg', '986.jpg', '181.jpg', '1266.jpg', '591.jpg', '1115.jpg', '537.jpg', '525.jpg', '1104.jpg', '1503.jpg', '1230.jpg', '1387.jpg', '172.jpg', '422.jpg', '665.jpg', '93.jpg', '984.jpg', '1534.jpg', '213.jpg', '1421.jpg', '1407.jpg', '540.jpg', '782.jpg', '312.jpg', '587.jpg', '950.jpg', '96.jpg', '1321.jpg', '586.jpg', '773.jpg', '735.jpg', '729.jpg', '117.jpg', '1126.jpg', '529.jpg', '226.jpg', '1086.jpg', '1433.jpg', '594.jpg', '602.jpg', '702.jpg', '940.jpg', '353.jpg', '1099.jpg', '236.jpg', '209.jpg', '1495.jpg', '1075.jpg', '205.jpg']
#generate data for all and save as CSV
all_imgs= ['547.jpg', '1139.jpg', '53.jpg', '1381.jpg', '659.jpg', '1459.jpg', '1028.jpg', '252.jpg', '962.jpg', '1263.jpg', '1152.jpg', '1217.jpg', '1000.jpg', '1542.jpg', '993.jpg', '1338.jpg', '406.jpg', '1438.jpg', '577.jpg', '858.jpg', '1150.jpg', '568.jpg', '845.jpg', '820.jpg', '1391.jpg', '423.jpg', '29.jpg', '1233.jpg', '1143.jpg', '310.jpg', '1227.jpg', '65.jpg', '630.jpg', '135.jpg', '87.jpg', '1335.jpg', '1294.jpg', '593.jpg', '224.jpg', '167.jpg', '333.jpg', '139.jpg', '674.jpg', '264.jpg', '1331.jpg', '245.jpg', '1214.jpg', '901.jpg', '84.jpg', '974.jpg', '1549.jpg', '942.jpg', '1536.jpg', '1047.jpg', '1473.jpg', '11.jpg', '1240.jpg', '711.jpg', '894.jpg', '164.jpg', '783.jpg', '362.jpg', '835.jpg', '926.jpg', '359.jpg', '918.jpg', '698.jpg', '504.jpg', '462.jpg', '413.jpg', '232.jpg', '1295.jpg', '1441.jpg', '1084.jpg', '655.jpg', '61.jpg', '304.jpg', '1372.jpg', '1014.jpg', '1370.jpg', '1468.jpg', '1033.jpg', '1158.jpg', '899.jpg', '919.jpg', '385.jpg', '138.jpg', '60.jpg', '417.jpg', '450.jpg', '985.jpg', '300.jpg', '46.jpg', '1502.jpg', '243.jpg', '1176.jpg', '818.jpg', '357.jpg', '614.jpg', '3.jpg', '103.jpg', '341.jpg', '1060.jpg', '596.jpg', '430.jpg', '237.jpg', '288.jpg', '18.jpg', '1485.jpg', '478.jpg', '875.jpg', '1501.jpg', '1332.jpg', '374.jpg', '588.jpg', '303.jpg', '1446.jpg', '1046.jpg', '947.jpg', '1121.jpg', '1259.jpg', '1061.jpg', '716.jpg', '269.jpg', '287.jpg', '12.jpg', '169.jpg', '1069.jpg', '520.jpg', '375.jpg', '1114.jpg', '965.jpg', '397.jpg', '747.jpg', '134.jpg', '648.jpg', '32.jpg', '1142.jpg', '1385.jpg', '1027.jpg', '627.jpg', '756.jpg', '714.jpg', '1484.jpg', '1524.jpg', '884.jpg', '255.jpg', '787.jpg', '1299.jpg', '1053.jpg', '809.jpg', '360.jpg', '1340.jpg', '416.jpg', '1302.jpg', '707.jpg', '16.jpg', '1322.jpg', '34.jpg', '250.jpg', '77.jpg', '584.jpg', '1095.jpg', '755.jpg', '730.jpg', '1467.jpg', '1565.jpg', '652.jpg', '634.jpg', '1109.jpg', '796.jpg', '1118.jpg', '1475.jpg', '1038.jpg', '1388.jpg', '1073.jpg', '45.jpg', '115.jpg', '931.jpg', '355.jpg', '113.jpg', '463.jpg', '1188.jpg', '1531.jpg', '1220.jpg', '1237.jpg', '458.jpg', '1540.jpg', '1568.jpg', '656.jpg', '1243.jpg', '1363.jpg', '1333.jpg', '273.jpg', '1244.jpg', '1103.jpg', '944.jpg', '500.jpg', '550.jpg', '1282.jpg', '492.jpg', '1545.jpg', '1051.jpg', '1491.jpg', '1254.jpg', '1092.jpg', '1415.jpg', '257.jpg', '271.jpg', '69.jpg', '1023.jpg', '1078.jpg', '1395.jpg', '633.jpg', '1377.jpg', '278.jpg', '177.jpg', '1523.jpg', '1380.jpg', '964.jpg', '127.jpg', '1490.jpg', '415.jpg', '380.jpg', '1396.jpg', '1113.jpg', '904.jpg', '372.jpg', '1245.jpg', '6.jpg', '952.jpg', '857.jpg', '866.jpg', '715.jpg', '200.jpg', '795.jpg', '681.jpg', '1177.jpg', '1369.jpg', '1454.jpg', '1519.jpg', '1174.jpg', '1500.jpg', '514.jpg', '363.jpg', '1179.jpg', '1290.jpg', '369.jpg', '814.jpg', '1032.jpg', '446.jpg', '1469.jpg', '1583.jpg', '1573.jpg', '1007.jpg', '1309.jpg', '708.jpg', '195.jpg', '68.jpg', '1168.jpg', '639.jpg', '258.jpg', '1339.jpg', '308.jpg', '238.jpg', '915.jpg', '1566.jpg', '1197.jpg', '1111.jpg', '609.jpg', '778.jpg', '71.jpg', '401.jpg', '1210.jpg', '179.jpg', '922.jpg', '578.jpg', '91.jpg', '1488.jpg', '571.jpg', '315.jpg', '524.jpg', '833.jpg', '1019.jpg', '1556.jpg', '234.jpg', '1580.jpg', '132.jpg', '1318.jpg', '912.jpg', '1248.jpg', '7.jpg', '1241.jpg', '1256.jpg', '1185.jpg', '402.jpg', '193.jpg', '686.jpg', '282.jpg', '1160.jpg', '229.jpg', '1234.jpg', '1492.jpg', '937.jpg', '507.jpg', '690.jpg', '533.jpg', '1352.jpg', '663.jpg', '1296.jpg', '343.jpg', '933.jpg', '1215.jpg', '1156.jpg', '546.jpg', '902.jpg', '1202.jpg', '666.jpg', '734.jpg', '22.jpg', '497.jpg', '545.jpg', '173.jpg', '1510.jpg', '654.jpg', '683.jpg', '322.jpg', '67.jpg', '757.jpg', '1212.jpg', '1293.jpg', '863.jpg', '1262.jpg', '1567.jpg', '1483.jpg', '853.jpg', '589.jpg', '761.jpg', '425.jpg', '760.jpg', '116.jpg', '1242.jpg', '121.jpg', '869.jpg', '936.jpg', '1462.jpg', '1224.jpg', '1530.jpg', '626.jpg', '1379.jpg', '1170.jpg', '1029.jpg', '791.jpg', '1079.jpg', '746.jpg', '934.jpg', '133.jpg', '5.jpg', '1098.jpg', '141.jpg', '274.jpg', '917.jpg', '805.jpg', '247.jpg', '1436.jpg', '272.jpg', '1199.jpg', '1107.jpg', '1432.jpg', '798.jpg', '15.jpg', '1541.jpg', '1328.jpg', '1430.jpg', '1394.jpg', '152.jpg', '903.jpg', '1102.jpg', '776.jpg', '911.jpg', '552.jpg', '154.jpg', '999.jpg', '995.jpg', '261.jpg', '231.jpg', '472.jpg', '1563.jpg', '1081.jpg', '1283.jpg', '324.jpg', '807.jpg', '1173.jpg', '1375.jpg', '466.jpg', '1110.jpg', '1329.jpg', '1089.jpg', '1498.jpg', '197.jpg', '145.jpg', '872.jpg', '834.jpg', '1466.jpg', '309.jpg', '1482.jpg', '339.jpg', '821.jpg', '959.jpg', '1516.jpg', '208.jpg', '488.jpg', '320.jpg', '684.jpg', '1271.jpg', '673.jpg', '739.jpg', '219.jpg', '62.jpg', '156.jpg', '1059.jpg', '214.jpg', '1159.jpg', '1376.jpg', '1552.jpg', '597.jpg', '802.jpg', '670.jpg', '168.jpg', '900.jpg', '1050.jpg', '1453.jpg', '486.jpg', '332.jpg', '556.jpg', '839.jpg', '543.jpg', '107.jpg', '482.jpg', '364.jpg', '432.jpg', '1068.jpg', '619.jpg', '28.jpg', '155.jpg', '1064.jpg', '298.jpg', '662.jpg', '1393.jpg', '862.jpg', '326.jpg', '590.jpg', '57.jpg', '1267.jpg', '1196.jpg', '680.jpg', '837.jpg', '847.jpg', '473.jpg', '351.jpg', '289.jpg', '1349.jpg', '1386.jpg', '600.jpg', '1140.jpg', '59.jpg', '828.jpg', '838.jpg', '82.jpg', '1246.jpg', '1274.jpg', '1247.jpg', '1304.jpg', '887.jpg', '1360.jpg', '1138.jpg', '418.jpg', '604.jpg', '1351.jpg', '535.jpg', '521.jpg', '1463.jpg', '867.jpg', '1.jpg', '198.jpg', '1544.jpg', '963.jpg', '560.jpg', '1024.jpg', '143.jpg', '361.jpg', '1133.jpg', '254.jpg', '972.jpg', '1494.jpg', '738.jpg', '775.jpg', '1258.jpg', '1423.jpg', '923.jpg', '354.jpg', '1058.jpg', '725.jpg', '1130.jpg', '94.jpg', '9.jpg', '956.jpg', '892.jpg', '1570.jpg', '561.jpg', '474.jpg', '1581.jpg', '749.jpg', '301.jpg', '551.jpg', '745.jpg', '508.jpg', '1305.jpg', '613.jpg', '885.jpg', '752.jpg', '852.jpg', '691.jpg', '1100.jpg', '1093.jpg', '563.jpg', '456.jpg', '1269.jpg', '1558.jpg', '485.jpg', '123.jpg', '1213.jpg', '967.jpg', '217.jpg', '1057.jpg', '576.jpg', '1087.jpg', '119.jpg', '2.jpg', '104.jpg', '779.jpg', '1105.jpg', '221.jpg', '453.jpg', '883.jpg', '1535.jpg', '408.jpg', '511.jpg', '1326.jpg', '280.jpg', '275.jpg', '1008.jpg', '1222.jpg', '822.jpg', '544.jpg', '1270.jpg', '412.jpg', '939.jpg', '1458.jpg', '292.jpg', '483.jpg', '951.jpg', '1319.jpg', '70.jpg', '1361.jpg', '898.jpg', '323.jpg', '108.jpg', '448.jpg', '476.jpg', '621.jpg', '801.jpg', '1507.jpg', '215.jpg', '1553.jpg', '871.jpg', '811.jpg', '759.jpg', '1091.jpg', '283.jpg', '438.jpg', '431.jpg', '218.jpg', '160.jpg', '74.jpg', '1307.jpg', '1548.jpg', '1280.jpg', '646.jpg', '1194.jpg', '204.jpg', '35.jpg', '1083.jpg', '1149.jpg', '36.jpg', '1003.jpg', '1561.jpg', '607.jpg', '223.jpg', '704.jpg', '1289.jpg', '846.jpg', '744.jpg', '291.jpg', '615.jpg', '792.jpg', '808.jpg', '207.jpg', '970.jpg', '979.jpg', '196.jpg', '701.jpg', '688.jpg', '1359.jpg', '1533.jpg', '348.jpg', '946.jpg', '1569.jpg', '941.jpg', '697.jpg', '443.jpg', '803.jpg', '618.jpg', '1281.jpg', '97.jpg', '957.jpg', '1341.jpg', '1203.jpg', '1410.jpg', '976.jpg', '342.jpg', '170.jpg', '503.jpg', '651.jpg', '89.jpg', '81.jpg', '1048.jpg', '1218.jpg', '1264.jpg', '713.jpg', '557.jpg', '1509.jpg', '1205.jpg', '334.jpg', '1412.jpg', '1443.jpg', '831.jpg', '157.jpg', '1417.jpg', '1279.jpg', '1356.jpg', '829.jpg', '527.jpg', '384.jpg', '23.jpg', '249.jpg', '506.jpg', '55.jpg', '882.jpg', '174.jpg', '1030.jpg', '1009.jpg', '978.jpg', '1182.jpg', '158.jpg', '1584.jpg', '945.jpg', '1055.jpg', '338.jpg', '650.jpg', '1255.jpg', '189.jpg', '608.jpg', '874.jpg', '1122.jpg', '677.jpg', '368.jpg', '1119.jpg', '1358.jpg', '743.jpg', '480.jpg', '788.jpg', '1010.jpg', '493.jpg', '1072.jpg', '1511.jpg', '1193.jpg', '592.jpg', '299.jpg', '1066.jpg', '628.jpg', '1250.jpg', '184.jpg', '248.jpg', '868.jpg', '162.jpg', '982.jpg', '344.jpg', '653.jpg', '518.jpg', '824.jpg', '183.jpg', '376.jpg', '1347.jpg', '80.jpg', '558.jpg', '1127.jpg', '1422.jpg', '953.jpg', '509.jpg', '1487.jpg', '137.jpg', '1232.jpg', '1464.jpg', '781.jpg', '433.jpg', '1314.jpg', '925.jpg', '1200.jpg', '1406.jpg', '1537.jpg', '1357.jpg', '751.jpg', '712.jpg', '909.jpg', '998.jpg', '733.jpg', '118.jpg', '1288.jpg', '210.jpg', '387.jpg', '336.jpg', '391.jpg', '1440.jpg', '4.jpg', '76.jpg', '335.jpg', '1559.jpg', '1137.jpg', '314.jpg', '836.jpg', '766.jpg', '21.jpg', '1532.jpg', '447.jpg', '1163.jpg', '1018.jpg', '105.jpg', '692.jpg', '92.jpg', '980.jpg', '244.jpg', '95.jpg', '1471.jpg', '1400.jpg', '122.jpg', '1367.jpg', '201.jpg', '467.jpg', '1315.jpg', '598.jpg', '1350.jpg', '736.jpg', '1261.jpg', '675.jpg', '48.jpg', '624.jpg', '1181.jpg', '699.jpg', '1192.jpg', '222.jpg', '832.jpg', '1366.jpg', '1090.jpg', '1187.jpg', '405.jpg', '30.jpg', '1116.jpg', '345.jpg', '1418.jpg', '1154.jpg', '799.jpg', '286.jpg', '328.jpg', '1353.jpg', '502.jpg', '717.jpg', '913.jpg', '719.jpg', '564.jpg', '523.jpg', '1317.jpg', '1228.jpg', '239.jpg', '490.jpg', '1011.jpg', '1550.jpg', '1209.jpg', '1547.jpg', '150.jpg', '327.jpg', '905.jpg', '794.jpg', '579.jpg', '192.jpg', '583.jpg', '649.jpg', '784.jpg', '921.jpg', '888.jpg', '955.jpg', '1354.jpg', '225.jpg', '1327.jpg', '559.jpg', '1062.jpg', '1571.jpg', '969.jpg', '1517.jpg', '1442.jpg', '1526.jpg', '41.jpg', '1300.jpg', '1165.jpg', '990.jpg', '1457.jpg', '307.jpg', '460.jpg', '966.jpg', '398.jpg', '166.jpg', '435.jpg', '762.jpg', '515.jpg', '1575.jpg', '1481.jpg', '434.jpg', '512.jpg', '549.jpg', '625.jpg', '907.jpg', '1508.jpg', '777.jpg', '227.jpg', '928.jpg', '1346.jpg', '242.jpg', '1088.jpg', '371.jpg', '14.jpg', '960.jpg', '949.jpg', '1522.jpg', '830.jpg', '1297.jpg', '228.jpg', '1486.jpg', '539.jpg', '370.jpg', '454.jpg', '1039.jpg', '1416.jpg', '349.jpg', '645.jpg', '1390.jpg', '457.jpg', '1424.jpg', '186.jpg', '1506.jpg', '1303.jpg', '1037.jpg', '532.jpg', '190.jpg', '530.jpg', '182.jpg', '253.jpg', '151.jpg', '120.jpg', '1239.jpg', '679.jpg', '50.jpg', '850.jpg', '1191.jpg', '373.jpg', '1470.jpg', '1265.jpg', '1404.jpg', '758.jpg', '1135.jpg', '1206.jpg', '347.jpg', '994.jpg', '706.jpg', '631.jpg', '484.jpg', '1198.jpg', '441.jpg', '1384.jpg', '570.jpg', '629.jpg', '1124.jpg', '1514.jpg', '851.jpg', '136.jpg', '1419.jpg', '1108.jpg', '565.jpg', '178.jpg', '891.jpg', '246.jpg', '1560.jpg', '114.jpg', '771.jpg', '147.jpg', '1136.jpg', '340.jpg', '1077.jpg', '436.jpg', '893.jpg', '1049.jpg', '606.jpg', '1252.jpg', '220.jpg', '973.jpg', '144.jpg', '1311.jpg', '191.jpg', '175.jpg', '536.jpg', '8.jpg', '331.jpg', '1439.jpg', '306.jpg', '1554.jpg', '260.jpg', '1521.jpg', '321.jpg', '1403.jpg', '426.jpg', '623.jpg', '1161.jpg', '268.jpg', '1451.jpg', '1171.jpg', '975.jpg', '38.jpg', '873.jpg', '111.jpg', '935.jpg', '88.jpg', '718.jpg', '1449.jpg', '188.jpg', '709.jpg', '748.jpg', '1431.jpg', '295.jpg', '1539.jpg', '420.jpg', '377.jpg', '797.jpg', '1183.jpg', '1420.jpg', '1022.jpg', '1278.jpg', '1112.jpg', '580.jpg', '1325.jpg', '763.jpg', '185.jpg', '1148.jpg', '800.jpg', '330.jpg', '620.jpg', '1374.jpg', '1131.jpg', '567.jpg', '109.jpg', '1146.jpg', '844.jpg', '1260.jpg', '1292.jpg', '1489.jpg', '1518.jpg', '386.jpg', '1447.jpg', '270.jpg', '554.jpg', '769.jpg', '1189.jpg', '616.jpg', '1426.jpg', '1231.jpg', '319.jpg', '1025.jpg', '1477.jpg', '13.jpg', '647.jpg', '881.jpg', '724.jpg', '531.jpg', '605.jpg', '522.jpg', '1013.jpg', '1277.jpg', '661.jpg', '1371.jpg', '265.jpg', '39.jpg', '510.jpg', '817.jpg', '789.jpg', '367.jpg', '641.jpg', '1450.jpg', '1310.jpg', '603.jpg', '678.jpg', '1557.jpg', '86.jpg', '279.jpg', '643.jpg', '1528.jpg', '658.jpg', '790.jpg', '128.jpg', '877.jpg', '981.jpg', '241.jpg', '861.jpg', '131.jpg', '421.jpg', '176.jpg', '958.jpg', '870.jpg', '569.jpg', '393.jpg', '1074.jpg', '98.jpg', '526.jpg', '311.jpg', '1287.jpg', '1582.jpg', '43.jpg', '496.jpg', '395.jpg', '566.jpg', '1236.jpg', '685.jpg', '1226.jpg', '1493.jpg', '1273.jpg', '263.jpg', '1414.jpg', '1579.jpg', '1368.jpg', '1362.jpg', '1094.jpg', '429.jpg', '924.jpg', '876.jpg', '1129.jpg', '827.jpg', '1141.jpg', '404.jpg', '705.jpg', '487.jpg', '750.jpg', '1020.jpg', '806.jpg', '1409.jpg', '638.jpg', '99.jpg', '230.jpg', '843.jpg', '1276.jpg', '394.jpg', '1225.jpg', '895.jpg', '720.jpg', '356.jpg', '1272.jpg', '1452.jpg', '610.jpg', '637.jpg', '1456.jpg', '622.jpg', '572.jpg', '73.jpg', '203.jpg', '1208.jpg', '992.jpg', '180.jpg', '445.jpg', '1320.jpg', '987.jpg', '1408.jpg', '1383.jpg', '1343.jpg', '396.jpg', '1425.jpg', '1063.jpg', '585.jpg', '1144.jpg', '1026.jpg', '1513.jpg', '1175.jpg', '854.jpg', '1162.jpg', '1336.jpg', '100.jpg', '469.jpg', '153.jpg', '1151.jpg', '1070.jpg', '1157.jpg', '316.jpg', '498.jpg', '943.jpg', '519.jpg', '1291.jpg', '1172.jpg', '352.jpg', '1045.jpg', '1427.jpg', '740.jpg', '52.jpg', '1080.jpg', '1562.jpg', '1313.jpg', '129.jpg', '305.jpg', '124.jpg', '810.jpg', '910.jpg', '859.jpg', '388.jpg', '1465.jpg', '31.jpg', '528.jpg', '977.jpg', '1238.jpg', '890.jpg', '541.jpg', '1455.jpg', '889.jpg', '1330.jpg', '1005.jpg', '284.jpg', '1401.jpg', '1190.jpg', '880.jpg', '489.jpg', '125.jpg', '1164.jpg', '403.jpg', '130.jpg', '860.jpg', '51.jpg', '159.jpg', '481.jpg', '285.jpg', '886.jpg', '26.jpg', '1128.jpg', '1308.jpg', '1397.jpg', '611.jpg', '676.jpg', '1085.jpg', '696.jpg', '856.jpg', '731.jpg', '1123.jpg', '826.jpg', '66.jpg', '961.jpg', '693.jpg', '199.jpg', '40.jpg', '1577.jpg', '1017.jpg', '948.jpg', '410.jpg', '455.jpg', '424.jpg', '276.jpg', '780.jpg', '664.jpg', '517.jpg', '470.jpg', '689.jpg', '848.jpg', '1479.jpg', '723.jpg', '281.jpg', '768.jpg', '212.jpg', '1578.jpg', '642.jpg', '1429.jpg', '669.jpg', '1155.jpg', '267.jpg', '37.jpg', '1125.jpg', '383.jpg', '1474.jpg', '1204.jpg', '879.jpg', '1520.jpg', '1201.jpg', '27.jpg', '419.jpg', '381.jpg', '841.jpg', '47.jpg', '1298.jpg', '914.jpg', '640.jpg', '772.jpg', '1478.jpg', '1021.jpg', '1076.jpg', '1551.jpg', '842.jpg', '816.jpg', '479.jpg', '1178.jpg', '1006.jpg', '741.jpg', '54.jpg', '1195.jpg', '991.jpg', '1106.jpg', '437.jpg', '617.jpg', '575.jpg', '1275.jpg', '1525.jpg', '534.jpg', '297.jpg', '896.jpg', '1445.jpg', '700.jpg', '163.jpg', '471.jpg', '1002.jpg', '1012.jpg', '211.jpg', '1052.jpg', '1036.jpg', '85.jpg', '1564.jpg', '825.jpg', '687.jpg', '767.jpg', '1147.jpg', '815.jpg', '259.jpg', '1040.jpg', '703.jpg', '235.jpg', '819.jpg', '1399.jpg', '1364.jpg', '240.jpg', '1348.jpg', '1546.jpg', '1413.jpg', '1497.jpg', '657.jpg', '1101.jpg', '251.jpg', '793.jpg', '112.jpg', '804.jpg', '101.jpg', '997.jpg', '1527.jpg', '337.jpg', '1249.jpg', '1082.jpg', '1512.jpg', '1223.jpg', '102.jpg', '671.jpg', '920.jpg', '442.jpg', '165.jpg', '582.jpg', '929.jpg', '414.jpg', '379.jpg', '1071.jpg', '1480.jpg', '1555.jpg', '595.jpg', '216.jpg', '574.jpg', '206.jpg', '202.jpg', '1323.jpg', '865.jpg', '477.jpg', '465.jpg', '1044.jpg', '411.jpg', '1134.jpg', '346.jpg', '409.jpg', '1132.jpg', '1444.jpg', '1411.jpg', '721.jpg', '44.jpg', '1221.jpg', '1435.jpg', '538.jpg', '78.jpg', '1389.jpg', '601.jpg', '542.jpg', '644.jpg', '1306.jpg', '72.jpg', '1437.jpg', '440.jpg', '897.jpg', '1344.jpg', '407.jpg', '317.jpg', '1043.jpg', '823.jpg', '840.jpg', '632.jpg', '1476.jpg', '390.jpg', '1056.jpg', '161.jpg', '878.jpg', '786.jpg', '294.jpg', '906.jpg', '553.jpg', '1515.jpg', '1015.jpg', '1499.jpg', '1180.jpg', '1382.jpg', '599.jpg', '1572.jpg', '499.jpg', '1216.jpg', '350.jpg', '495.jpg', '612.jpg', '296.jpg', '233.jpg', '266.jpg', '399.jpg', '695.jpg', '505.jpg', '849.jpg', '1316.jpg', '989.jpg', '770.jpg', '318.jpg', '58.jpg', '366.jpg', '1355.jpg', '555.jpg', '171.jpg', '1145.jpg', '682.jpg', '1286.jpg', '365.jpg', '672.jpg', '1402.jpg', '732.jpg', '1065.jpg', '1405.jpg', '20.jpg', '1342.jpg', '392.jpg', '1251.jpg', '1285.jpg', '1334.jpg', '277.jpg', '996.jpg', '668.jpg', '660.jpg', '313.jpg', '126.jpg', '1324.jpg', '33.jpg', '256.jpg', '110.jpg', '1219.jpg', '1268.jpg', '728.jpg', '1054.jpg', '694.jpg', '464.jpg', '1505.jpg', '428.jpg', '753.jpg', '1392.jpg', '187.jpg', '63.jpg', '146.jpg', '1576.jpg', '400.jpg', '908.jpg', '494.jpg', '389.jpg', '988.jpg', '19.jpg', '1472.jpg', '1207.jpg', '79.jpg', '1117.jpg', '378.jpg', '1067.jpg', '1538.jpg', '513.jpg', '194.jpg', '1016.jpg', '954.jpg', '459.jpg', '75.jpg', '64.jpg', '25.jpg', '930.jpg', '439.jpg', '90.jpg', '1398.jpg', '764.jpg', '1035.jpg', '24.jpg', '983.jpg', '1257.jpg', '1428.jpg', '1434.jpg', '293.jpg', '1365.jpg', '916.jpg', '83.jpg', '562.jpg', '986.jpg', '181.jpg', '1166.jpg', '1266.jpg', '1461.jpg', '1543.jpg', '106.jpg', '591.jpg', '1337.jpg', '1115.jpg', '537.jpg', '525.jpg', '548.jpg', '1104.jpg', '1235.jpg', '501.jpg', '1503.jpg', '1284.jpg', '142.jpg', '1230.jpg', '1097.jpg', '1387.jpg', '172.jpg', '1167.jpg', '754.jpg', '422.jpg', '302.jpg', '665.jpg', '667.jpg', '1042.jpg', '93.jpg', '573.jpg', '984.jpg', '1184.jpg', '1534.jpg', '475.jpg', '1004.jpg', '636.jpg', '1186.jpg', '581.jpg', '1496.jpg', '213.jpg', '1301.jpg', '1378.jpg', '290.jpg', '1421.jpg', '726.jpg', '148.jpg', '1407.jpg', '1169.jpg', '329.jpg', '540.jpg', '1001.jpg', '1120.jpg', '782.jpg', '312.jpg', '927.jpg', '449.jpg', '1312.jpg', '710.jpg', '587.jpg', '950.jpg', '461.jpg', '96.jpg', '727.jpg', '1321.jpg', '42.jpg', '586.jpg', '737.jpg', '932.jpg', '968.jpg', '864.jpg', '773.jpg', '1034.jpg', '444.jpg', '735.jpg', '358.jpg', '729.jpg', '1574.jpg', '1345.jpg', '17.jpg', '813.jpg', '635.jpg', '117.jpg', '10.jpg', '516.jpg', '1126.jpg', '1448.jpg', '529.jpg', '325.jpg', '1041.jpg', '1460.jpg', '774.jpg', '1253.jpg', '226.jpg', '722.jpg', '1229.jpg', '382.jpg', '1086.jpg', '1433.jpg', '427.jpg', '594.jpg', '971.jpg', '1529.jpg', '602.jpg', '765.jpg', '702.jpg', '742.jpg', '452.jpg', '855.jpg', '940.jpg', '353.jpg', '1099.jpg', '491.jpg', '262.jpg', '236.jpg', '1031.jpg', '938.jpg', '451.jpg', '1373.jpg', '209.jpg', '49.jpg', '1495.jpg', '1504.jpg', '1153.jpg', '140.jpg', '812.jpg', '1211.jpg', '149.jpg', '1075.jpg', '468.jpg', '785.jpg', '56.jpg', '1096.jpg', '205.jpg']
#test_sample_images = ['547.jpg', '1139.jpg', '53.jpg', '1381.jpg', '659.jpg', '1459.jpg', '1028.jpg', '252.jpg']

#run openCV analysis on all images and return dataframe of results
df = runEval(all_imgs)

#output OpenCV data to all_image_output_v2.csv
df.to_csv('all_image_output_v2.csv', sep='\t')
