# Light Detector
An extended version of PyImageSearch "Detecting multiple bright spots in an image with Python and OpenCV"
https://www.pyimagesearch.com/2016/10/31/detecting-multiple-bright-spots-in-an-image-with-python-and-opencv/


```shell
python light_detector_cli.py --help
usage: light_detector_cli.py [-h] -i IMAGE [--minthresh MINTHRESH]
                             [--maxthresh MAXTHRESH] [--outpath OUTPATH]
                             [--json] [--visualize] [--visualout]
                             [--infillout] [--maskout]
                             [--iterations ITERATIONS]

optional arguments:
  -h, --help            show this help message and exit
  -i IMAGE, --image IMAGE
                        Path to the input image file
  --minthresh MINTHRESH
                        Min light threshold value
  --maxthresh MAXTHRESH
                        Max light threshold value
  --outpath OUTPATH     Directory output path to save files
  --json                Output light info to json file
  --visualize           Visualize light detection
  --visualout           Output light detection visualization
  --infillout           Output image with lights filled from surrounding
                        pixels
  --maskout             Output of light mask
  --iterations ITERATIONS
                        Iterations to apply for in-fill process

```