import numpy as np
import argparse
import math
import detect_head
import sys
import cv2
from sklearn.cluster import estimate_bandwidth
from collections import OrderedDict
import pandas as pd

# global variables
MIN_DISTANCE = 6
GROUP_DISTANCE = 15
COLOR_POOL = [[1,50,200],[150,200,9],[255,0,0],[10,79,0],[200,100,0],[25,38,0],[78,250,250],[158,143,120],[28,222,98],[255,250,199],[100,150,100],[200,50,1],[250,250,78],[15,38,245],[123,234,90]]

def parse_arguments():
	# construct the argument parser and parse the arguments
	ap = argparse.ArgumentParser()
	ap.add_argument("-p", "--shape-predictor", required=True,
		help="path to facial landmark predictor")
	ap.add_argument("-i", "--image", required=True,
		help="path to input image")
	args = vars(ap.parse_args())
	return args

def euclidean_distance(x1, x2):
	dis = (sum((x1[:2] - x2[:2])**2))*1.5
	color = sum((x1[2:] - x2[2:])**2)
	#return np.sqrt(sum((x1-x2)**2))
	return np.sqrt(dis + color)

# formula: (1/(sigma*sqrt(2*pi)))*e^(-x^2/(2*sigma^2))
def gaussian(distance, kernel):
	return (1/(kernel*math.sqrt(2*math.pi)))*np.exp(-0.5*((distance/kernel))**2)

# the weighted mean of the density in the window determined by the Gaussian kernel
# (sum of gaussian(distance)*point) / (sum of gaussian(distance))
def move(center_point, points, kernel):
	sum_x = 0
	sum_y = 0
	sum_b = 0
	sum_g = 0
	sum_r = 0
	sum_weight = 0

	for p in points:
		distance = euclidean_distance(p, center_point)
		weight = gaussian(distance, kernel)
		sum_x += weight * p[0]
		sum_y += weight * p[1]
		sum_b += weight * p[2]
		sum_g += weight * p[3]
		sum_r += weight * p[4]
		sum_weight += weight

	return np.array([(sum_x / sum_weight), (sum_y / sum_weight), (sum_b / sum_weight), (sum_g / sum_weight), (sum_r / sum_weight)])

def get_neighbors(all_points, center_point, kernel):
	neighbors = []
	for p in all_points:
		distance = euclidean_distance(p, center_point)
		if distance <= kernel:
			neighbors.append(p)
	return neighbors

def find_min_dis(group, pos):
	min_distance = sys.float_info.max
	for p in group:
		min_distance = min(min_distance, euclidean_distance(p, pos));
	return min_distance

def group_clusters(positions, points):
	#print("positions are: ", positions)
	#print("points are: ", points)
	# label each original points
	labels = [0]
	# a list to group the positions, for the purpose of calculating the min distance
	grouped_positions = [[positions[0]]]
	# a list to group the original points
	# once we know the lable of the original point, we can quickly find other elements in the same group
	grouped_points = [np.array([points[0]])]
	total_index = 1
	for i in range(1, len(positions)):
		print("processing the {}th element".format(i))
		need_new_index = True
		for index, group in enumerate(grouped_positions):
			#print("the group is: ", group)
			#print("the point is: ", positions[i])
			#print("min distance is ", find_min_dis(group, positions[i]))
			#print("len of grouped_positions", len(grouped_positions))
			if find_min_dis(group, positions[i]) <= GROUP_DISTANCE:
				labels.append(index)
				grouped_positions[index].append(positions[i])
				grouped_points[index] = np.vstack([grouped_points[index], points[i]])
				need_new_index = False
				break
		if need_new_index:
			#print("new index############################################")
			labels.append(total_index)
			grouped_positions.append([positions[i]])
			grouped_points.append(np.array([points[i]]))
			total_index += 1

	'''
	with open('positions.txt', 'w') as f:
		f.write(str(grouped_positions))

	with open('points.txt', 'w') as f:
		f.write(str(grouped_points))
	'''
	print("grouped_points is ", grouped_points)
	dd1=pd.DataFrame(grouped_positions)
	dd2=pd.DataFrame(grouped_points)
	dd1.to_csv("pos.csv")
	dd2.to_csv("pts.csv")
	return labels, np.array(grouped_points)
	

def mean_shift(points, kernel):
	# we mark a point as visited when it is converged(measured by distance)
	# type: {tuple: boolean}
	visited = OrderedDict((tuple(p), False) for p in points)
	#print(visited)

	# mapping from the original points to points after shifted
	# type: {tuple: ndarray}
	original_to_shifted = OrderedDict((tuple(p), p) for p in points)
	#print(original_to_shifted)

	converged = False
	iteration = 0																										

	# if not all points have converged
	while not converged or iteration > 30:
		converged = True
		#print("iteration #####################################", iteration)
		for original_point in original_to_shifted:
			#print("original point", original_point)
			cur_pos = original_to_shifted[original_point]
			#print("cur pos", cur_pos)
			if visited[tuple(cur_pos)]:
				#print("an element has finished.")
				continue
			neighbors = get_neighbors(points, cur_pos, kernel) # revise
			#print("len of neighbors, ", len(neighbors))
			next_pos = move(cur_pos, neighbors, kernel) # revise
			#print("next pos, ", next_pos)
			original_to_shifted[original_point] = next_pos
			distance_dif = euclidean_distance(cur_pos, next_pos)
			#print("distance dif, ", distance_dif)
			if distance_dif <= MIN_DISTANCE:
				visited[tuple(next_pos)] = True
			else:
				visited[tuple(next_pos)] = False
				converged = False
		iteration += 1
		#print('\n')
	#print("total iterations: ", iteration)

	#with open('out.txt', 'w') as f:
	#	f.write(str(iteration))

	#with open('out.txt', 'a') as f:
	#	f.write(str(original_to_shifted))

	positions = list(original_to_shifted.values())
	dd=pd.DataFrame(original_to_shifted)
	dd.to_csv("original_to_shifted.csv")
	#print(positions)
	#print(points)
	# after all ponits are converge d, we group and label the clusters
	labels, grouped_points = group_clusters(positions, points)
	return labels, grouped_points

def segmentation():
	args = parse_arguments()
	image, points, width, height = detect_head.get_points(args)
	#interested_region = np.reshape(np.array(interested_region), [-1,3])
	#print(interested_region.shape)
	#kernel = estimate_bandwidth(np.asnp.array(interested_region), quantile=0.1, n_samples=len(points))
	#print(kernel)
	#print(len(points))
	labels, shifted_points = mean_shift(points, 15)
	#print(labels)
	#print(shifted_points)

	'''
	# segment hair
	# 1. get initial point class: bg_left, bg_right
	x_height = image.shape[0] * 1/3
	y_left = 0
	y_right = image.shape[1] - 1
	x_height_line = []
	bg_left = -1
	bg_right = -1

	print("labels are: ", labels)
	for index, point in enumerate(points):
		print("point[x] is", point[0])
		print("point[y] is", point[1])
		if point[0] ==x_height and point[1] == y_left:
			bg_left = labels[index]
		if point[0] == x_height and point[1] == y_right:
			bg_right = labels[index]
		if point[0] == x_height:
			x_height_line.append(index)
		if point[0] > x_height:
			break

	# 2. get hair class: hair_class
	print("the x_height_line is: ", x_height_line)
	left = 0
	right = len(x_height_line) - 1
	hair_class = -1
	while left < right:
		if labels[x_height_line[left]] != bg_left and labels[x_height_line[right]] != bg_right and labels[x_height_line[left]] == labels[x_height_line[right]]:
			hair_class =  labels[x_height_line[left]]
			break
		else:
			left += 1
			right -= 1

	# 3. segment out hair class
	print("the hair class is: ", hair_class)
	print("the bg_left class is: ", bg_left)
	print("the bg_right class is: ", bg_right)
	hair_members = []
	for index, label in enumerate(labels):
		if label == hair_class:
			hair_members.append(points[index])

	print("hairs are: ", hair_members)
	# draw on the image
	index = 0
	'''
	
	print(len(shifted_points))
	print(len(shifted_points))
	print(len(shifted_points))
	print(len(shifted_points))
	print(len(shifted_points))
	

	# create blank image
	# blank_image = np.zeros((image.shape[0],image.shape[1],3), np.uint8)

	#dd=pd.DataFrame(hair_members)
	#dd.to_csv("hair_members.csv")
	index = 0
	for points in shifted_points:
		'''
		points = np.array(points)
		cv2.circle(image, (x, y), 1, (0, 0, 255), -1)
		arr = np.array(points[:,:2])
		print(arr)
		hull = cv2.convexHull(arr)
		cv2.drawContours(image, [hull], -1, COLOR_POOL[index], -1)
		'''
		if len(points) > 50:
			for point in points:
				x = point[1]
				y = point[0]
				print(index)
				color = COLOR_POOL[index][::-1]
				cv2.circle(image, (x, y), 1, color, -1)
			index += 1
		
		'''
		x = h[1]
		y = h[0]
		cv2.circle(blank_image, (x, y), 1, COLOR_POOL[index], -1)
		'''

	
	image = cv2.resize(image,(width,height),interpolation=cv2.INTER_CUBIC)
	cv2.putText(image, "MIN_DISTANCE:{}, GROUP_DISTANCE:{}, KERNAL:15".format(MIN_DISTANCE, GROUP_DISTANCE), (200, 0), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
	cv2.imwrite("Output.jpg", image)
	
	'''
	hair_image = cv2.resize(blank_image,(width,height),interpolation=cv2.INTER_CUBIC)
	cv2.imwrite("hair.jpg", hair_image)
	'''
	# hair edge line detection


	return shifted_points

segmentation()