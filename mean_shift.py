import numpy as np
import argparse
import math
import sys
import cv2
from collections import OrderedDict
import pandas as pd
import scipy.io as sio

# global variables
MIN_DISTANCE = 5
GROUP_DISTANCE = 10
COLOR_POOL = [[1,50,200],[150,200,9],[255,0,0],[10,79,0],\
			  [200,100,0],[25,38,0],[78,250,250],[158,143,120],[28,222,98],\
			  [255,250,199],[100,150,100],[200,50,1],[250,250,78],[15,38,245],\
			  [123,234,90]]


def preprocess(args):
	# load the input image, resize it, and convert it to grayscale
	image = cv2.imread(args["image"])
	original_height = int(image.shape[0])
	original_width = int(image.shape[1])
	image = cv2.resize(image,(50,50),interpolation=cv2.INTER_CUBIC)
	all_points = []

	for i in range(0, image.shape[0]):
		for j in range(0, image.shape[1]):
			all_points.append(np.array([i,j] + list(image[i,j]))) # revised

	return image, all_points, original_width, original_height

def euclidean_distance(x1, x2):
	# weighted on the distance more
	dis = (sum((x1[:2] - x2[:2])**2))*1.5
	color = sum((x1[2:] - x2[2:])**2)
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

	return np.array([(sum_x / sum_weight), (sum_y / sum_weight), \
		(sum_b / sum_weight), (sum_g / sum_weight), (sum_r / sum_weight)])

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
			if find_min_dis(group, positions[i]) <= GROUP_DISTANCE:
				labels.append(index)
				grouped_positions[index].append(positions[i])
				grouped_points[index] = np.vstack([grouped_points[index], points[i]])
				need_new_index = False
				break
		if need_new_index:
			labels.append(total_index)
			grouped_positions.append([positions[i]])
			grouped_points.append(np.array([points[i]]))
			total_index += 1

	return labels, np.array(grouped_points)
	

def mean_shift(points, kernel):
	# we mark a point as visited when it is converged(measured by distance)
	# type: {tuple: boolean}
	visited = OrderedDict((tuple(p), False) for p in points)

	# mapping from the original points to points after shifted
	# type: {tuple: ndarray}
	original_to_shifted = OrderedDict((tuple(p), p) for p in points)

	converged = False
	iteration = 0																										

	# if not all points have converged
	while not converged or iteration > 30:
		converged = True
		for original_point in original_to_shifted:
			cur_pos = original_to_shifted[original_point]
			print(cur_pos)
			if visited[tuple(cur_pos)]:
				continue
			neighbors = get_neighbors(points, cur_pos, kernel) # revise
			next_pos = move(cur_pos, neighbors, kernel) # revise
			original_to_shifted[original_point] = next_pos
			distance_dif = euclidean_distance(cur_pos, next_pos)
			if distance_dif <= MIN_DISTANCE:
				visited[tuple(next_pos)] = True
			else:
				visited[tuple(next_pos)] = False
				converged = False
		iteration += 1

	positions = list(original_to_shifted.values())

	# after all ponits are converge d, we group and label the clusters
	labels, grouped_points = group_clusters(positions, points)

	return labels, grouped_points

def segmentation(args):
	image, points, width, height = preprocess(args)
	labels, shifted_points = mean_shift(points, 15)

	index = 0
	# draw the result of the segmentation
	for pts in shifted_points:
		if len(pts) > 50:
			for point in pts:
				x = point[1]
				y = point[0]
				color = COLOR_POOL[index][::-1]
				cv2.circle(image, (x, y), 1, color, -1)
			index += 1
	
	image_resized = cv2.resize(image,(width,height),interpolation=cv2.INTER_CUBIC)
	cv2.imwrite("Segmentation.jpg", image_resized)
	return labels, shifted_points, image, points, width, height