
from roboflow import Roboflow
rf = Roboflow(api_key="fuJLc06TfhMuwNrrRjcF")
project = rf.workspace("suranaree-university-xjb3n").project("illegal-helmet-detection")
dataset = project.version(8).download("yolov8")