import json
import os
import cv2

f = json.load(open('ITSDGroundTruth.json'))

labelToidx = {'mandatory': 0, 'cautionary': 1, 'informatory': 2, 'other': 3}
true = dict()

for i in f:
	name = i['filename']
	nme = './ITSD_annotation/'+name+'.txt'
	wr = open(nme, 'w')
	true[name] = list()
	H, W = 1080, 1920

	for j in i['objects']:
		l=list()
		cx = j['relative_coordinates']['center_x']
		cy = j['relative_coordinates']['center_y']
		h = j['relative_coordinates']['height']
		w = j['relative_coordinates']['width']
		x_min = (cx)*W
		y_min = (cy)*H
		x_max = (cx+w)*W
		y_max = (cy+h)*H
		cx = (x_max+x_min)/(2*W)
		cy = (y_max+y_min)/(2*H)

		l.extend([int(x_min), int(y_min), int(x_max), int(y_max)])
		true[name].append(l)

		label = j['name'].split('-')[0]
		wr.write(f"{labelToidx[label]} {cx} {cy} {w} {h}\n")
		# print(name, cx, cy, h, w, label)
	wr.close()
	
# f = json.load(open('ITSDGroundTruth2.json'))

# for i in f:
# 	name = i['filename']
# 	nme = './India_images_annotation/'+name+'.txt'
# 	wr = open(nme, 'w')
# 	for j in i['objects']:
# 		cx = j['relative_coordinates']['center_x']
# 		cy = j['relative_coordinates']['center_y']
# 		h = j['relative_coordinates']['height']
# 		w = j['relative_coordinates']['width']
# 		label = j['name'].split('-')[0]
# 		wr.write(f"{labelToidx[label]} {cx} {cy} {w} {h}\n")
# 		# print(name, cx, cy, h, w, label)
# 	wr.close()

# f = json.load(open('result_final.json'))

# for i in f:
# 	name = i['filename'].split('/')[-1]
# 	nme = './Traffic images_annotation/'+name+'.txt'
# 	wr = open(nme, 'w')
# 	for j in i['objects']:
# 		cx = j['relative_coordinates']['center_x']
# 		cy = j['relative_coordinates']['center_y']
# 		h = j['relative_coordinates']['height']
# 		w = j['relative_coordinates']['width']
# 		label = j['name'].split('-')[0]
# 		wr.write(f"{labelToidx[label]} {cx} {cy} {w} {h}\n")
# 		# print(name, cx, cy, h, w, label)
# 	wr.close()


for name in os.listdir('./itsd/'):
		print(name) 
		# name = "Screenshot (326).png"
		path = "/home/sundesh/Documents/git/btp/miten/itsd/"
		img = cv2.imread(path+name)
		print(img.shape)
		# print(pred[name])
		print(true[name])
		# for i in range(len(pred[name])):
			# x = cv2.rectangle(img, (pred[name][i][0], pred[name][i][1]), (pred[name][i][2], pred[name][i][3]), (255, 0, 0), 2)
		for i in range(len(true[name])):
			x = cv2.rectangle(img, (true[name][i][0], true[name][i][1]), (true[name][i][2], true[name][i][3]), (0, 255, 0), 2)	
		# x = cv2.rectangle(img, (true[name][0][0], true[name][0][1]), (true[name][0][2], true[name][0][3]), (255, 0, 0), 2)
		# show(img, name)
		cv2.imwrite(name, img)
	#######################