import numpy as np
import sys
import cv2
import math
import os

DATA_DIR = "."
os.makedirs('./processed', exist_ok = True)
SAVE_DIR = './processed/'

def show(img, title = "Image"):
	cv2.imshow(title, img)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

def get_corners(bboxes):
	
	"""Get corners of bounding boxes
	
	Parameters
	----------
	
	bboxes: numpy.ndarray
		Numpy array containing bounding boxes of shape `N X 4` where N is the 
		number of bounding boxes and the bounding boxes are represented in the
		format `x1 y1 x2 y2`
	
	returns
	-------
	
	numpy.ndarray
		Numpy array of shape `N x 8` containing N bounding boxes each described by their 
		corner co-ordinates `x1 y1 x2 y2 x3 y3 x4 y4`      
		
	"""
	width = (bboxes[:,2] - bboxes[:,0]).reshape(-1,1)
	height = (bboxes[:,3] - bboxes[:,1]).reshape(-1,1)
	
	x1 = bboxes[:,0].reshape(-1,1)
	y1 = bboxes[:,1].reshape(-1,1)
	
	x2 = x1 + width
	y2 = y1 
	
	x3 = x1
	y3 = y1 + height
	
	x4 = bboxes[:,2].reshape(-1,1)
	y4 = bboxes[:,3].reshape(-1,1)
	
	corners = np.hstack((x1,y1,x2,y2,x3,y3,x4,y4))
	
	return corners

def rotate_box(corners,angle,  cx, cy, h, w):
	
	"""Rotate the bounding box.
	
	
	Parameters
	----------
	
	corners : numpy.ndarray
		Numpy array of shape `N x 8` containing N bounding boxes each described by their 
		corner co-ordinates `x1 y1 x2 y2 x3 y3 x4 y4`
	
	angle : float
		angle by which the image is to be rotated
		
	cx : int
		x coordinate of the center of image (about which the box will be rotated)
		
	cy : int
		y coordinate of the center of image (about which the box will be rotated)
		
	h : int 
		height of the image
		
	w : int 
		width of the image
	
	Returns
	-------
	
	numpy.ndarray
		Numpy array of shape `N x 8` containing N rotated bounding boxes each described by their 
		corner co-ordinates `x1 y1 x2 y2 x3 y3 x4 y4`
	"""

	corners = corners.reshape(-1,2)
	corners = np.hstack((corners, np.ones((corners.shape[0],1), dtype = type(corners[0][0]))))
	
	M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
	
	
	cos = np.abs(M[0, 0])
	sin = np.abs(M[0, 1])
	
	nW = int((h * sin) + (w * cos))
	nH = int((h * cos) + (w * sin))
	# adjust the rotation matrix to take into account translation
	M[0, 2] += (nW / 2) - cx
	M[1, 2] += (nH / 2) - cy
	# Prepare the vector to be transformed
	calculated = np.dot(M,corners.T).T
	
	calculated = calculated.reshape(-1,8)
	
	return calculated

def get_enclosing_box(corners):
	"""Get an enclosing box for ratated corners of a bounding box
	
	Parameters
	----------
	
	corners : numpy.ndarray
		Numpy array of shape `N x 8` containing N bounding boxes each described by their 
		corner co-ordinates `x1 y1 x2 y2 x3 y3 x4 y4`  
	
	Returns 
	-------
	
	numpy.ndarray
		Numpy array containing enclosing bounding boxes of shape `N X 4` where N is the 
		number of bounding boxes and the bounding boxes are represented in the
		format `x1 y1 x2 y2`
		
	"""
	x_ = corners[:,[0,2,4,6]]
	y_ = corners[:,[1,3,5,7]]
	
	xmin = np.min(x_,1).reshape(-1,1)
	ymin = np.min(y_,1).reshape(-1,1)
	xmax = np.max(x_,1).reshape(-1,1)
	ymax = np.max(y_,1).reshape(-1,1)
	
	final = np.hstack((xmin, ymin, xmax, ymax,corners[:,8:]))
	
	return final

def rotate_im(image, angle):
	"""Rotate the image.
	
	Rotate the image such that the rotated image is enclosed inside the tightest
	rectangle. The area not occupied by the pixels of the original image is colored
	black. 
	
	Parameters
	----------
	
	image : numpy.ndarray
		numpy image
	
	angle : float
		angle by which the image is to be rotated
	
	Returns
	-------
	
	numpy.ndarray
		Rotated Image
	
	"""
	# grab the dimensions of the image and then determine the
	# centre
	(h, w) = image.shape[:2]
	(cX, cY) = (w // 2, h // 2)

	# grab the rotation matrix (applying the negative of the
	# angle to rotate clockwise), then grab the sine and cosine
	# (i.e., the rotation components of the matrix)
	M = cv2.getRotationMatrix2D((cX, cY), angle, 1.0)
	cos = np.abs(M[0, 0])
	sin = np.abs(M[0, 1])

	# compute the new bounding dimensions of the image
	nW = int((h * sin) + (w * cos))
	nH = int((h * cos) + (w * sin))

	# adjust the rotation matrix to take into account translation
	M[0, 2] += (nW / 2) - cX
	M[1, 2] += (nH / 2) - cY

	# perform the actual rotation and return the image
	image = cv2.warpAffine(image, M, (nW, nH))

#    image = cv2.resize(image, (w,h))
	return image

def process(l, angle):
	filename = l[0]     
	img_path = os.path.join(DATA_DIR, filename)
	for i in range(1, len(l)):
		l[i] = int(l[i])
	img = cv2.imread(img_path)

	# disp = cv2.rectangle(img, (l[1], l[2]), (l[3], l[4]), color = (0, 0, 0), thickness = 3)
	# show(disp)

	w,h = img.shape[1], img.shape[0]
	cx, cy = w//2, h//2
	# perform the actual rotation and return the image
	rotated_img = rotate_im(img, angle)

	bboxes = np.array(l[1:5]);
	bboxes = bboxes.reshape(1, 4)

	corners = get_corners(bboxes)
	corners = np.hstack((corners, bboxes[:,4:]))
	corners[:,:8] = rotate_box(corners[:,:8], angle, cx, cy, h, w)
	new_bbox = get_enclosing_box(corners).astype('float')
	scale_factor_x = rotated_img.shape[1] / w

	scale_factor_y = rotated_img.shape[0] / h

	rotated_img = cv2.resize(rotated_img, (w,h))

	new_bbox[:,:4] /= [scale_factor_x, scale_factor_y, scale_factor_x, scale_factor_y] 

	new_bbox = new_bbox.astype('int')

	# disp = cv2.rectangle(rotated_img, (new_bbox[0][0], new_bbox[0][1]), (new_bbox[0][2], new_bbox[0][3]), color = (0, 0, 0), thickness = 3)
	# show(rotated_img)

	img_name = filename.split('.')[0] + "_" + str(angle) + "." + filename.split('.')[1]
	new_l = [img_name, str(new_bbox[0][0]), str(new_bbox[0][1]), str(new_bbox[0][2]), str(new_bbox[0][3]), str(l[5])]
	return new_l, rotated_img

if __name__ == '__main__':
	txt = open('gt.txt', 'r')
	new_txt = open('gt_new.txt', 'w')
	cnt = 0
	for line in txt.readlines():
		if (cnt == 0):
			new_txt.write(line)
			cnt += 1
			continue
		l = line.split(';')
		for angle in range(-20, 20, 5):
			new_l, img = process(l, angle)
			new_line = ';'.join(new_l) + "\n"

			cv2.imwrite(os.path.join(SAVE_DIR, new_l[0]), img)
			new_txt.write(new_line)
			cnt += 1
