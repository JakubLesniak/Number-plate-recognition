import numpy as np
import cv2
import imutils
from os import listdir
from os.path import isfile, join
import argparse
from pathlib import Path

def detectPlate(image, lowThresh):
	image = imutils.resize(image, width=500)
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	gray = cv2.bilateralFilter(gray, 11, 17, 17)
	ret, thresh = cv2.threshold(gray, lowThresh, 255, cv2.THRESH_BINARY)
	return thresh

def contains(bb, bb_list):
	for bb2 in bb_list:
		x,y,w,h = bb
		x2,y2,w2,h2 = bb2
		if x > x2 and w < w2 and y > y2 and h < h2 and x2+w2 > x+w and y2+h2 > y+h:
			return True
	return False
	
def cropChars(img):
	contours, _  = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

	bb_list = []
	for c in contours:  
		bb = cv2.boundingRect(c)
		if (cv2.contourArea(c) < 350 or bb[2]*bb[3] > img.shape[0]*img.shape[1]/4 or bb[2] > bb[3]):
			continue
		bb_list.append(bb)
	bb_list = [bb for bb in bb_list if not contains(bb, bb_list)]
	return [img[y:y+h, x:x+w] for x,y,w,h in sorted(bb_list, key=lambda x: x[0])]
	
def trainModel():
	imagePaths = [f for f in listdir('./data') if isfile(join('./data', f))]
	data = [(path.split('_')[0], cv2.resize(cv2.cvtColor(cv2.imread('./data/'+path), cv2.COLOR_BGR2GRAY), (100,200))) for path in imagePaths]
	train = np.array([d[1] for d in data]).reshape(-1,20000).astype(np.float32)
	train_labels = np.array([[ord(d[0])] for d in data])
	knn = cv2.ml.KNearest_create()
	knn.train(train,cv2.ml.ROW_SAMPLE,train_labels)
	return knn
	
def getChar(knn, img):
	i = cv2.resize(img, (100,200)).reshape(-1,20000).astype(np.float32)
	_,result,_,_ = knn.findNearest(i,k=5)
	return chr(np.array(int(result)))

def detectChars(path, knn):
	image = cv2.imread(path)

	chars = []
	lowThresh = 250
	while len(chars) != 7 and lowThresh > 0:
		plate = detectPlate(image, lowThresh)
		if plate is not None:
			chars = cropChars(plate)
		lowThresh -= 10
	return '"' + path + '": "' + ''.join([getChar(knn, img) for img in chars]) + '"'

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('images_dir', type=str)
	parser.add_argument('results_file', type=str)
	args = parser.parse_args()

	images_dir = Path(args.images_dir)
	results_file = Path(args.results_file)

	images_paths = sorted([str(image_path) for image_path in images_dir.iterdir() if image_path.name.endswith('.jpg')])

	knn = trainModel()
	with results_file.open('w') as output_file:
		output_file.write('{\n\t' + ',\n\t'.join([detectChars(f, knn) for f in images_paths]) + '\n}')


if __name__ == '__main__':
	main()
