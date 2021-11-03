# import the necessary packages
from imutils import contours
from skimage import measure
import numpy as np
import quaternion
from astropy import coordinates
import argparse
import imutils
import cv2
import pprint
import json


def write_json(lightsdict, filepath):
    with open(filepath, 'w') as outfile:
        json.dump(lightsdict, outfile, indent=4)


def coords_to_degrees(coordx, coordy, width, height):
    degreex = coordx/width * 360 - 180
    degreey = coordy/height * 180 - 90
    return [degreex, degreey]


def degrees_to_rads(degree):
    rad = degree * (np.pi/180)
    return rad


# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()

ap.add_argument("-i", "--image",
                required=True,
                help="path to the image file")

ap.add_argument("--minthresh",
                type=float,
                default=1.0,
                help="min threshold value")

ap.add_argument("--maxthresh",
                type=float,
                default=10000,
                help="max threshold value")

ap.add_argument("--outpath",
                type=str,
                help="output path to save yaml file")

ap.add_argument("--visualize",
                action='store_true',
                default=False,
                help="visualize light detection")

args = ap.parse_args()

# load the image, convert it to grayscale, and blur it
image = cv2.imread(args.image, cv2.IMREAD_UNCHANGED)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (11, 11), 0)
print("[INFO] Image Shape: {}".format(image.shape))

# threshold the image to reveal light regions in the
# blurred image
thresh = cv2.threshold(blurred, args.minthresh, args.maxthresh, cv2.THRESH_BINARY)[1]

# perform a series of erosions and dilations to remove
# any small blobs of noise from the thresholded image
thresh = cv2.erode(thresh, None, iterations=2)
thresh = cv2.dilate(thresh, None, iterations=4)

# perform a connected component analysis on the thresholded
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

# find the contours in the mask, then sort them from left to
# right
cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
                        cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
cnts = contours.sort_contours(cnts)[0]
print("[INFO] Detected {} Light Sources".format(len(cnts)))

lightsdict = dict()
lightsdict['path'] = args.image
lightsdict['lights'] = dict()
lightsdict['shape'] = image.shape

# loop over the contours
for (i, c) in enumerate(cnts):

    # setup dict to store lights info
    lightkey = "light_{}".format(str(i+1).zfill(4))
    lightsdict['lights'][lightkey] = dict()

    # draw the bright spot on the image
    (x, y, w, h) = cv2.boundingRect(c)
    lightsdict['lights'][lightkey]['bounds'] = (x,y,w,h)
    lightsdict['lights'][lightkey]['coverage'] = w / image.shape[1], h / image.shape[0]
    lightsdict['lights'][lightkey]['center'] = x + w/2, y + h/2

    # coonvert coords to cartesian
    degrees = coords_to_degrees(lightsdict['lights'][lightkey]['center'][0],
                                lightsdict['lights'][lightkey]['center'][1],
                                lightsdict['shape'][1],
                                lightsdict['shape'][0])
    radianx, radiany = degrees_to_rads(degrees[0]), degrees_to_rads(degrees[1])
    cartesian = coordinates.spherical_to_cartesian(1.0, radiany, radianx)
    lightsdict['lights'][lightkey]['cartesian'] = float(cartesian[0]),\
                                                  float(cartesian[1]),\
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
    print(colornorm)
    cv2.circle(image, (int(cX), int(cY)), int(radius),
               (colornorm[0], colornorm[1], colornorm[2]), int(image.shape[1]/500))
    cv2.putText(image, "{}".format(lightkey), (x, y - 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=image.shape[1]/2000,
                color=(colornorm[0], colornorm[1], colornorm[2]),
                thickness=int(image.shape[1]/500))

# show the output image
if args.visualize:
    cv2.imshow("Image", image)
    cv2.waitKey(0)

print(pprint.pprint(lightsdict))

if args.outpath:
    write_json(lightsdict, args.outpath)
