import json
import cv2
import numpy as np
import pickle

np.set_printoptions(precision=3, suppress=True)


a_file = open("shapeData.pkl", "rb")
shapes = pickle.load(a_file)

def show(img, title = "Image"):
	cv2.imshow(title, img)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

def process_itsd():
	json_content = json.load(open('../miten/result_final.json'))
	json_content1 = json.load(open('../miten/ITSDGroundTruth2.json'))
	json_content3 = json.load(open('../miten/ITSDGroundTruth.json'))

	D = dict()

	name2idx = dict()
	cnt = 0

	for x in json_content:
		name = x['filename'].split('/')[-1]
		H, W = shapes[name][0], shapes[name][1]

		D[name] = list()
		for val in x['objects']:
			l = list()
			cx = val['relative_coordinates']['center_x']
			cy = val['relative_coordinates']['center_y']
			w = val['relative_coordinates']['width']
			h = val['relative_coordinates']['height']
			x_min = (cx)*W
			y_min = (cy)*H
			x_max = (cx+w)*W
			y_max = (cy+h)*H
			l.extend([int(x_min), int(y_min), int(x_max), int(y_max)])

			sign_name = val['name'].split('-')[0]

			if sign_name not in name2idx:
				name2idx[sign_name] = cnt
				cnt += 1

			l.append(name2idx[sign_name])

			D[name].append(l)

	for x in json_content1:
		name = x['filename'].split('/')[-1]

		if name not in shapes:
			continue

		H, W = shapes[name][0], shapes[name][1]
		D[name] = list()
		for val in x['objects']:
			l = list()
			cx = val['relative_coordinates']['center_x']
			cy = val['relative_coordinates']['center_y']
			w = val['relative_coordinates']['width']
			h = val['relative_coordinates']['height']
			x_min = (cx)*W
			y_min = (cy)*H
			x_max = (cx+w)*W
			y_max = (cy+h)*H
			l.extend([int(x_min), int(y_min), int(x_max), int(y_max)])

			sign_name = val['name'].split('-')[0]

			if sign_name not in name2idx:
				name2idx[sign_name] = cnt
				cnt += 1

			l.append(name2idx[sign_name])

			D[name].append(l)

	for x in json_content3:
		name = x['filename'].split('/')[-1]
		H, W = shapes[name][0], shapes[name][1]
		D[name] = list()
		for val in x['objects']:
			l = list()
			cx = val['relative_coordinates']['center_x']
			cy = val['relative_coordinates']['center_y']
			w = val['relative_coordinates']['width']
			h = val['relative_coordinates']['height']
			x_min = (cx)*W
			y_min = (cy)*H
			x_max = (cx+w)*W
			y_max = (cy+h)*H
			l.extend([int(x_min), int(y_min), int(x_max), int(y_max)])

			sign_name = val['name'].split('-')[0]

			if sign_name not in name2idx:
				name2idx[sign_name] = cnt
				cnt += 1

			l.append(name2idx[sign_name])

			D[name].append(l)

	print(name2idx)
	return D



def process_itsd_result():

# json_content = json.load(open('itsd_result_original.json'))

# 	Recall   : 0.7907128842380641
# 	Precision: 0.7845554834523037
# 	F1       : 0.7876221498371335
# 	Accuracy : 0.6496507254164428

	# json_content = json.load(open('itsd_result_original_nighttime.json'))

# Recall   : 0.7704381948986265
# Precision: 0.7714472822527833
# F1       : 0.7709424083769634
# Accuracy : 0.6272630457933972

	# json_content = json.load(open('itsd_result_l4_v1.json'))

# Recall   : 0.8077174623937214
# Precision: 0.7670807453416149
# F1       : 0.7868748008920037
# Accuracy : 0.6486344537815126

	# json_content = json.load(open('itsd_result_l4_v1_nighttime.json'))

# Recall   : 0.7979071288423807
# Precision: 0.7731305449936628
# F1       : 0.7853234631477309
# Accuracy : 0.6465288818229995


	D = dict()

	name2idx = dict()
	cnt = 0

	for x in json_content:
		name = x['filename'].split('/')[-1]
		H, W = shapes[name][0], shapes[name][1]
		D[name] = list()
		for val in x['objects']:
			l = list()
			cx = val['relative_coordinates']['center_x']
			cy = val['relative_coordinates']['center_y']
			w = val['relative_coordinates']['width']
			h = val['relative_coordinates']['height']
			x_min = (cx-w/2)*W
			y_min = (cy-h/2)*H
			x_max = (cx+w/2)*W
			y_max = (cy+h/2)*H
			l.extend([int(x_min), int(y_min), int(x_max), int(y_max)])
			l.append(val['class_id'])

			sign_name = val['name'].split('-')[0]

			if sign_name not in name2idx:
				name2idx[sign_name] = cnt
				cnt += 1

			D[name].append(l)

	print(name2idx)
	return D

# def detection_accuracy(true, pred):
# 	detection_accuracy(true, pred)
# 	cnt = 0	
# 	for key, val in pred.items():
# 		t_len = len(true[key])
# 		p_len = len(val)

# 		if (len(val) == 0):
# 			cnt += 1
# 	print(cnt)

def bbox2dict(bbox):
	d = dict()
	d['x1'] = bbox[0]
	d['y1'] = bbox[1]
	d['x2'] = bbox[2]
	d['y2'] = bbox[3]
	return d

def get_iou(bb1, bb2):
    """
    Calculate the Intersection over Union (IoU) of two bounding boxes.

    Parameters
    ----------
    bb1 : dict
        Keys: {'x1', 'x2', 'y1', 'y2'}
        The (x1, y1) position is at the top left corner,
        the (x2, y2) position is at the bottom right corner
    bb2 : dict
        Keys: {'x1', 'x2', 'y1', 'y2'}
        The (x, y) position is at the top left corner,
        the (x2, y2) position is at the bottom right corner

    Returns
    -------
    float
        in [0, 1]
    """

    assert bb1['x1'] < bb1['x2']
    assert bb1['y1'] < bb1['y2']
    assert bb2['x1'] < bb2['x2']
    assert bb2['y1'] < bb2['y2']

    # determine the coordinates of the intersection rectangle
    x_left = max(bb1['x1'], bb2['x1'])
    y_top = max(bb1['y1'], bb2['y1'])
    x_right = min(bb1['x2'], bb2['x2'])
    y_bottom = min(bb1['y2'], bb2['y2'])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # The intersection of two axis-aligned bounding boxes is always an
    # axis-aligned bounding box
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # compute the area of both AABBs
    bb1_area = (bb1['x2'] - bb1['x1']) * (bb1['y2'] - bb1['y1'])
    bb2_area = (bb2['x2'] - bb2['x1']) * (bb2['y2'] - bb2['y1'])

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
    assert iou >= 0.0
    assert iou <= 1.0
    return iou

def get_mapping(true, pred, classwise = 0, iou_thresh = None):

	mapping = np.zeros((4, 4))
	tp = 0
	fp = 0
	fn = 0

	zero_ious = dict()
	not_found = 0
	for key, val in pred.items():
		fp += len(val)
	for key, val in true.items():
		if key not in pred:
			not_found += 1
			continue
		p = pred[key]
		# fp += len(p)
		fn += len(val)
		for bbox_true in val:
			pred_class = -1
			mx_iou = 0
			# original += 1	# tp + fn
			for bbox_pred in p:
				c_iou = get_iou(bbox2dict(bbox_true[:-1]), bbox2dict(bbox_pred[:-1]))
				# print(f"ciou: {c_iou}, true_id: {bbox_true[-1]}, pred_id: {bbox_pred[-1]}")
				# mapping[bbox_true[-1], bbox_pred[-1]] += c_iou
				if (c_iou > mx_iou):
					mx_iou = c_iou
					pred_class = bbox_pred[-1]
			if classwise:
				if mx_iou >= iou_thresh:
					tp += 1	
			else:
				if mx_iou != 0:
					# print(mx_iou)
					mapping[bbox_true[-1], pred_class] += 1
					tp += 1
				else:
					true_class = bbox_true[-1]
					if true_class not in zero_ious:
						zero_ious[true_class] = 0
					zero_ious[true_class] += 1
			# print('-'*10)
	fp -= tp
	fn -= tp
	# print(mapping)
	print("not found:", not_found)
	print("zero ious:", zero_ious)
	print(f"tp: {tp}, fp: {fp}, fn: {fn}")

	print(f"Recall   : {tp/(tp+fn)}")
	print(f"Precision: {tp/(tp+fp)}")
	precision = tp/(tp+fp)
	recall = tp/(tp+fn)
	print(f"F1       : {2*precision*recall/(precision+recall)}")
	print(f"Accuracy : {tp/(tp+fp+fn)}")

	return mapping

if __name__ == '__main__':
	true = process_itsd()
	print(len(true))

	counter = 0
	for key, val in true.items():
		# print(key, val)
		counter += len(val)
	# 	break
	# exit(0)
	print("scd:", counter)

	pred = process_itsd_result()
	print(len(pred))

	# for key, val in pred.items():
	# 	print(key, val)
	# 	break

	mapping = get_mapping(true, pred)

	print(mapping)
	true2pred = np.argmax(mapping, axis = 1)

	print(true2pred)
	# p0, p1, t0, t1 = dict(), dict(), dict(), dict()
	# cnt2 = 0
	# cnt3 = 0

	# for key, val in pred.items():
	# 	p0[key] = list()
	# 	p1[key] = list()
	# 	for output in val:
	# 		if (output[-1] == 0):
	# 			p0[key].append(output)
	# 		if (output[-1] == 1):
	# 			p1[key].append(output)
	# 		if (output[-1] == 2):
	# 			cnt2 += 1
	# 		if (output[-1] == 3):
	# 			cnt3 += 1

	# print(f"Number of predictions of class 2: {cnt2}")
	# print(f"Number of predictions of class 3: {cnt3}")

	# for key, val in true.items():
	# 	t0[key] = list()
	# 	t1[key] = list()
	# 	for output in val:
	# 		if (true2pred[output[-1]] == 0):
	# 			t0[key].append(output)
	# 		if (true2pred[output[-1]] == 1):
	# 			t1[key].append(output)

	# print("class 0:")
	# get_mapping(t0, p0, classwise = 1, iou_thresh = 0.5)
	# print("class 1:")
	# get_mapping(t1, p1, classwise = 1, iou_thresh = 0.5)

	for key, val in true.items():
		filename = './mAP/input/ground-truth/'+key+'.txt'
		f = open(filename, 'w')
		for l in val:
			l1 = [true2pred[l[-1]]]
			l1.extend(l[:-1])
			s = " ".join([ str(j) for j in l1])
			s += "\n"
			f.write(s)
		f.close()

	for key, val in pred.items():
		filename = './mAP/input/detection-results/'+key+'.txt'
		f = open(filename, 'w')
		for l in val:
			# if (l[-1] == 0 or l[-1] == 1 or l[-1] == 2):
			l1 = [l[-1], 1]
			l1.extend(l[:-1])
			s = " ".join([ str(j) for j in l1])
			s += "\n"
			f.write(s)
		f.close()

	## testing bounding box
	import os

	for name in os.listdir('./itsd/'):
		# name = "Screenshot (326).png"
		path = "/home/sundesh/Documents/git/btp/cross_dataset/itsd/"
		img = cv2.imread(path+name)
		name = name.split('.')[0] + '.jpg'
		# print(img.shape)
		# print(pred[name])
		# print(true[name])
		if name not in pred or name not in true:
			continue
		print(name) 

		if name in pred:
			for i in range(len(pred[name])):
				x = cv2.rectangle(img, (pred[name][i][0], pred[name][i][1]), (pred[name][i][2], pred[name][i][3]), (255, 0, 0), 2)
		if name in true:
			for i in range(len(true[name])):
				x = cv2.rectangle(img, (true[name][i][0], true[name][i][1]), (true[name][i][2], true[name][i][3]), (0, 255, 0), 2)	
		# x = cv2.rectangle(img, (true[name][0][0], true[name][0][1]), (true[name][0][2], true[name][0][3]), (255, 0, 0), 2)
		# show(img, name)
		cv2.imwrite(name, img)
	#######################

	
	
	

