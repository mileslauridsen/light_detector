import os
import imutils
from imutils import contours
from skimage import measure
import numpy as np
from astropy import coordinates
import math
import argparse
import cv2
import json
import datetime
import logging

# set logging
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)
logfile = "light_detector_{}.log".format(datetime.datetime.now().strftime("%Y%m%d"))
output_file_handler = logging.FileHandler(logfile)
log.addHandler(output_file_handler)

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()

ap.add_argument("-i", "--image",
                required=True,
                help="Path to the input image file")

ap.add_argument("--minthresh",
                type=float,
                default=1.0,
                help="Min light threshold value")

ap.add_argument("--maxthresh",
                type=float,
                default=10000,
                help="Max light threshold value")

ap.add_argument("--outpath",
                type=str,
                help="Directory output path to save files")

ap.add_argument("--json",
                action='store_true',
                default=False,
                help="Output light info to json file")

ap.add_argument("--visualize",
                action='store_true',
                default=False,
                help="Visualize light detection")

ap.add_argument("--visualout",
                action='store_true',
                default=False,
                help="Output light detection visualization")

ap.add_argument("--infillout",
                action='store_true',
                default=False,
                help="Output image with lights filled from surrounding pixels")

ap.add_argument("--maskout",
                action='store_true',
                default=False,
                help="Output of light mask")

ap.add_argument("--iterations",
                type=int,
                default=8,
                help="Iterations to apply for in-fill process")

arguments = ap.parse_args()


def write_json(lightsdict, filepath):
	"""
	Write dict of lights info to file
	:param lightsdict: dict of lights
	:param filepath: path string
    :return: None
    """
	if os.path.isdir(os.path.dirname(filepath)):
		with open(filepath, 'w') as outfile:
			json.dump(lightsdict, outfile, indent=4)


def coords_to_degrees(coordx, coordy, width, height):
	"""
	Convert 2d coords to degrees
    :param coordx: float x coordinate
    :param coordy: float y coordinate
    :param width: float of image width
    :param height: float of image height
    :return: horizontal and vertical degree values as floats
    """
	degreex = coordx / width * 360 - 180
	degreey = coordy / height * 180 - 90
	return [degreex, degreey]


def degrees_to_rads(degreex, degreey):
	"""
    Converts degrees to radians for Nuke cartesian
	:param degreex: float degree
    :param degreey: float degree
    :return: horizontal and vertical radian values as floats
    """
	radianx = (degreex - 90) * (np.pi / 180)
	radiany = (90 - degreey) * (np.pi / 180)
	return radianx, radiany


def degrees_to_rads2(degreex, degreey):
	"""
	Standard degree to radian conversion
	:param degreex: float degree x
	:param degreey: float degree y
	:return: Horizontal and vertical radian values as floats
	"""
	radianx = math.radians(degreex)
	radiany = math.radians(degreey)
	return radianx, radiany


def polar2cart(r, theta, phi):
	"""
	Convert polar radians to cartesian coords
    :param r: float radius
    :param theta: float polar angle
    :param phi: float azimuthal angle
    :return: array of cartesian coords, y up
    """
	return [
		r * np.sin(theta) * np.cos(phi),
		r * np.cos(theta),
		r * np.sin(theta) * np.sin(phi)
	]


def detector(args):
	"""
	Main detector script
	"""
	# load the image, convert it to grayscale, and blur it
	imagepath = args.image
	if os.path.isfile(imagepath):
		log.info("Detecting Lights In {}".format(imagepath))
		image = cv2.imread(imagepath, cv2.IMREAD_UNCHANGED)
		gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
		blurred = cv2.GaussianBlur(gray, (11, 11), 0)
		log.info("Image Size: {}x{}".format(image.shape[1], image.shape[0]))

	# copy image for imgfill and add full alpha
	if args.infillout:
		imgfill = cv2.cvtColor(image.copy(), cv2.COLOR_RGB2RGBA)

	# threshold the image to reveal light regions in the
	# blurred image
	thresh = cv2.threshold(blurred, args.minthresh, args.maxthresh, cv2.THRESH_BINARY)[1]

	# perform a series of erosions and dilations to remove
	# any small blobs of noise from the thresholded image
	thresh = cv2.erode(thresh, None, iterations=2)
	thresh = cv2.dilate(thresh, None, iterations=4)

	# perform a connected component analysis on the threshold
	# image, then initialize a mask to store only the "large"
	# components
	labels = measure.label(thresh, background=0)
	mask = np.zeros(thresh.shape, dtype="uint8")
	# loop over the unique components
	for label in np.unique(labels):
		# if this is the background label, ignore it
		if label == 0:
			continue
		# otherwise, construct the label mask and count the
		# number of pixels
		labelMask = np.zeros(thresh.shape, dtype="uint8")
		labelMask[labels == label] = 255
		numPixels = cv2.countNonZero(labelMask)
		# if the number of pixels in the component is sufficiently
		# large, then add it to our mask of "large blobs"
		if numPixels > 300:
			mask = cv2.add(mask, labelMask)

	# find the contours in the mask, then sort them from left to right
	cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
	                        cv2.CHAIN_APPROX_SIMPLE)
	cnts = imutils.grab_contours(cnts)
	cnts = contours.sort_contours(cnts)[0]
	log.info("Detected {} Light Sources".format(len(cnts)))

	# remove lights from image
	if args.infillout:
		# setup mask as alpha
		alpha_imgfill = cv2.bitwise_not(mask)

		# Stencil out light mask
		imgfill = cv2.bitwise_and(imgfill, imgfill, mask=alpha_imgfill)
		imgfill[:, :, 3] = np.array(alpha_imgfill / 255.0, dtype="float32")
		imgdownsize = imgfill.copy()

		downscale = 0.5

		for i in range(args.iterations):
			imgdownsize = cv2.resize(imgdownsize,
			                         (int(imgdownsize.shape[1] * downscale), int(imgdownsize.shape[0] * downscale)))
			for color in range(0, 4):
				fixnans = np.array(imgdownsize[:, :, color])
				fixnans = np.nan_to_num(fixnans)
				imgdownsize[:, :, color] = fixnans

			imgupsize = cv2.resize(imgdownsize,
			                       (image.shape[1], image.shape[0]),
			                       interpolation=cv2.INTER_LINEAR)
			for color in range(0, 4):
				imgupsize[:, :, color] = cv2.divide(imgupsize[:, :, color], imgupsize[:, :, 3])
				fixnans = np.array(imgupsize[:, :, color])
				fixnans = np.nan_to_num(fixnans)
				imgupsize[:, :, color] = fixnans
				imgupsize[:, :, color] = cv2.blur(imgupsize[:, :, color],
				                                  (int(image.shape[1] / 100), int(image.shape[1] / 100)))

			# set adjusted colors
			for color in range(0, 3):
				imgfill[:, :, color] = cv2.add((1.0 - imgfill[:, :, 3]) * imgupsize[:, :, color], imgfill[:, :, color])

			# max the alphas
			imgfill[:, :, 3] = cv2.max(imgfill[:, :, 3], imgupsize[:, :, 3])

		imgfill_blur = cv2.blur(imgfill, (50, 50))
		# set adjusted colors
		for color in range(0, 3):
			imgfill[:, :, color] = cv2.add(cv2.blur((1.0 - np.array(alpha_imgfill / 255.0, dtype="float32")),
			                                        (int(image.shape[1] / 100),
			                                         int(image.shape[1] / 100))) * imgfill_blur[:, :, color],
			                               imgfill[:, :, color] * cv2.blur(
				                               np.array(alpha_imgfill / 255.0, dtype="float32"),
				                               (int(image.shape[1] / 100), int(image.shape[1] / 100))))

	lightsdict = dict()
	lightsdict['path'] = args.image
	lightsdict['lights'] = dict()
	lightsdict['shape'] = image.shape

	# loop over the contours
	for (i, c) in enumerate(cnts):

		# setup dict to store lights info
		lightkey = "light_{}".format(str(i + 1).zfill(4))
		lightsdict['lights'][lightkey] = dict()

		# draw the bright spot on the image
		(x, y, w, h) = cv2.boundingRect(c)
		lightsdict['lights'][lightkey]['bounds'] = (x, y, w, h)
		lightsdict['lights'][lightkey]['coverage'] = w / image.shape[1], h / image.shape[0]
		lightsdict['lights'][lightkey]['center'] = x + w / 2, y + h / 2

		# convert coords to cartesian
		degrees = coords_to_degrees(float(lightsdict['lights'][lightkey]['center'][0]),
		                            float(lightsdict['lights'][lightkey]['center'][1]),
		                            float(lightsdict['shape'][1]),
		                            float(lightsdict['shape'][0]))

		radianx, radiany = degrees_to_rads(degrees[0]), degrees_to_rads(degrees[1])
		cartesian = coordinates.spherical_to_cartesian(1.0, radiany, radianx)
		lightsdict['lights'][lightkey]['cartesian'] = float(cartesian[0]), \
		                                              float(cartesian[1]), \
		                                              float(cartesian[2])

		# measure light info for given area
		lightselect = image[y:y + h, x:x + w]
		channels = ("b", "g", "r")
		coordstd = []
		coordmax = []
		coordmedian = []
		coordavg = []
		for chan in channels:
			coordstd.append(float(np.std(lightselect[:, :, channels.index(chan)])))
			coordmax.append(float(np.max(lightselect[:, :, channels.index(chan)])))
			coordmedian.append(float(np.median(lightselect[:, :, channels.index(chan)])))
			coordavg.append(float(np.average(lightselect[:, :, channels.index(chan)])))

		lightsdict['lights'][lightkey]['stddev'] = coordstd
		lightsdict['lights'][lightkey]['max'] = coordmax
		lightsdict['lights'][lightkey]['median'] = coordmedian
		lightsdict['lights'][lightkey]['average'] = coordavg

		# draw the circle
		((cX, cY), radius) = cv2.minEnclosingCircle(c)
		colormax = max(coordavg)
		colornorm = [(x / colormax) for x in coordavg]
		cv2.circle(image, (int(cX), int(cY)), int(radius),
		           (colornorm[0], colornorm[1], colornorm[2]), int(image.shape[1] / 500))
		cv2.putText(image, "{}".format(lightkey), (x, y - 15),
		            cv2.FONT_HERSHEY_SIMPLEX,
		            fontScale=image.shape[1] / 2000,
		            color=(colornorm[0], colornorm[1], colornorm[2]),
		            thickness=int(image.shape[1] / 500))

	# show the output image
	if args.visualize:
		cv2.imshow("Image", image)
		cv2.waitKey(0)

	# output infill image
	if args.infillout:
		if os.path.isdir(args.outpath):
			filename = os.path.join(args.outpath,
			                        "{}_{}.exr".format(os.path.splitext(os.path.basename(args.image))[0],
			                                           "lights_infill"))
			cv2.imwrite(filename, imgfill)
			log.info("Infill Output: {}".format(filename))

	# output lights mask as default
	if args.maskout:
		if os.path.isdir(args.outpath):
			filename = os.path.join(args.outpath,
			                        "{}_{}.png".format(os.path.splitext(os.path.basename(args.image))[0],
			                                           "lights_mask"))
			cv2.imwrite(filename, mask)
			log.info("Mask Output: {}".format(filename))

	# output visualized image
	if args.visualout:
		if os.path.isdir(args.outpath):
			filename = os.path.join(args.outpath,
			                        "{}_{}.exr".format(os.path.splitext(os.path.basename(args.image))[0],
			                                           "lights_visualized"))
			cv2.imwrite(filename, image)
			log.info("Infill Output: {}".format(filename))

	# output json of lights info
	if args.json:
		if os.path.isdir(args.outpath):
			filename = os.path.join(args.outpath,
			                        "{}_{}.json".format(os.path.splitext(os.path.basename(args.image))[0],
			                                            "lights"))
			write_json(lightsdict, filename)
			log.info("Lights JSON Output: {}".format(filename))


if __name__ == "__main__":
	detector(arguments)
