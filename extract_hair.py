import nunmpy as np
import math
import mean_shift

def hair_line():
	# segment hair
	labels, shifted_points, image, points, width, height = mean_shift.segmentation()

	# 1. get initial point class: bg_left, bg_right
	x_height = image.shape[0] * 1/3
	y_left = 0
	y_right = image.shape[1] - 1
	x_height_line = []
	bg_left = -1
	bg_right = -1

	for index, point in enumerate(points):
		if point[0] ==x_height and point[1] == y_left:
			bg_left = labels[index]
		if point[0] == x_height and point[1] == y_right:
			bg_right = labels[index]
		if point[0] == x_height:
			x_height_line.append(index)
		if point[0] > x_height:
			break

	# 2. get hair class: hair_class
	left = 0
	right = len(x_height_line) - 1
	hair_class = -1
	while left < right:
		if labels[x_height_line[left]] != bg_left:
			hair_class =  labels[x_height_line[left]]
			break
		else:
			left += 1

	# 3. segment out hair class
	hair_members = []
	for index, label in enumerate(labels):
		if label == hair_class:
			hair_members.append(points[index])

	# 4. get hair edge
	hair_line = OrderedDict()
	for h in hair_members:
		if h[0] in hair_line:
			hair_line[h[0]].append(h[1])
		else:
			hair_line[h[0]] = [h[1]]

	result = []
	index = len(hair_line)
	for h in hair_line:
		if index <= 2:
			break
		index -= 1
		hair_line[h].sort()
		result.append([h, hair_line[h][0]])
		result.append([h, hair_line[h][-1]])
		if len(hair_line[h]) > 2:
			print len(hair_line[h])
			for i in range(len(hair_line[h]) - 1, 0, -1):
				if hair_line[h][i] - hair_line[h][i-1] > 1:
					result.append([h, hair_line[h][i]])
					break
			for i in range(0, len(hair_line[h]) - 1):
				if hair_line[h][i+1] - hair_line[h][i] > 1:
					result.append([h, hair_line[h][i]])
					break

	#create blank image
	blank_image = np.zeros((width,height,3), np.uint8)
	blank_image.fill(255)

	ratio_w = width / image.shape[1]
	ratio_h = height / image.shape[0]
	result = [np.array([r[0]*ratio_h*1.1, (r[1]*ratio_w-25)*1.1]) for r in result]

	for r in result:
		x = r[1]
		y = r[0]
		cv2.circle(blank_image, (x, y), 1, COLOR_POOL[0])

	# save for mathlab use
	result = np.array(result)
	sio.savemat('points.mat', {'obj_arr':result})  

	# write to hair edge line image
	cv2.imwrite("edge.jpg", blank_image)


hair_line()