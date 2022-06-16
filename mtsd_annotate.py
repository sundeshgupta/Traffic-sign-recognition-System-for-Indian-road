import json

f = open('train.txt')

s = dict()
idx = 0

for filename in f.readlines():

	d = json.load(open(filename[:-1]+'.json'))


	for objects in d['objects']:
		if objects['label'] not in s:
			s[objects['label']] = idx
			idx += 1

f.close()
f = open('train.txt')
for filename in f.readlines():
	d = json.load(open(filename[:-1]+'.json'))

	h = d['height']
	w = d['width']

	nme = './train/'+filename[:-1]+'.txt'
	wr = open(nme, 'w')
	for objects in d['objects']:
		xmin = objects['bbox']['xmin']
		xmax = objects['bbox']['xmax']
		ymax = objects['bbox']['ymax']
		ymin = objects['bbox']['ymin']
		cx = (xmin+xmax)/(2*w)
		cy = (ymin+ymax)/(2*h)
		width = (xmax-xmin)/w
		height = (ymax-ymin)/w
		wr.write(f"{s[objects['label']]} {cx} {cy} {width} {height}")
		wr.write("\n")

	wr.close()

f = open('val.txt')
for filename in f.readlines():
	d = json.load(open(filename[:-1]+'.json'))

	h = d['height']
	w = d['width']

	nme = './val/'+filename+'.txt'
	wr = open(nme, 'w')
	for objects in d['objects']:
		xmin = objects['bbox']['xmin']
		xmax = objects['bbox']['xmax']
		ymax = objects['bbox']['ymax']
		ymin = objects['bbox']['ymin']
		cx = (xmin+xmax)/(2*w)
		cy = (ymin+ymax)/(2*h)
		width = (xmax-xmin)/w
		height = (ymax-ymin)/w
		wr.write(f"{s[objects['label']]} {cx} {cy} {width} {height}")
		wr.write("\n")

	wr.close()




