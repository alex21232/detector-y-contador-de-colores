# Color Detection and Object Counting System

## Introduction
This project features two Python scripts for real-time color detection and object counting using computer vision. The first script demonstrates color segmentation in LAB color space, while the second implements more advanced object detection in HSV space with morphological operations and contour analysis. Both programs use OpenCV to process live video from a webcam.

## Table of Contents
| Section | Description |
|---------|-------------|
| [Script 1: LAB Color Detection](#script-1-lab-color-detection) | Simple color segmentation in LAB space |
| [Script 2: HSV Object Counter](#script-2-hsv-object-counter) | Advanced object detection with counting |
| [Dependencies](#dependencies) | Required libraries |
| [Resources](#resources) | Learning materials |
| [Conclusion](#conclusion) | Project summary |
| [Author](#author) | Creator information |

## Script Details

### Script 1: LAB Color Detection
| Feature | Description |
|---------|-------------|
| Color Spaces | Uses LAB color space for segmentation |
| Colors Detected | Black, Blue, Green, Red, Purple |
| Output | Separate windows for each color mask |
| Controls | Press 'q' to quit |

### Script 2: HSV Object Counter
| Feature | Description |
|---------|-------------|
| Color Spaces | Uses HSV color space |
| Processing | Includes morphological operations (erosion/dilation) |
| Detection | Finds and counts colored objects |
| Visualization | Draws contours, centroids, and labels |
| Metrics | Displays real-time object counts |
| Controls | Press 'q' to quit |

## Dependencies
- OpenCV (`opencv-python`)
- NumPy (`numpy`)

Install with:
```bash
pip install opencv-python numpy
```

## Resources
- [OpenCV Color Conversions](https://docs.opencv.org/4.x/d8/d01/group__imgproc__color__conversions.html)
- [HSV Color Space Explained](https://learnopencv.com/color-spaces-in-opencv-cpp-python/)
- [Morphological Operations Guide](https://docs.opencv.org/4.x/d9/d61/tutorial_py_morphological_ops.html)
- [Contour Features Documentation](https://docs.opencv.org/4.x/dd/d49/tutorial_py_contour_features.html)

## Conclusion
These scripts demonstrate different approaches to color-based object detection. The LAB version provides simple color segmentation, while the HSV implementation offers complete object detection with counting capabilities. The project serves as an excellent foundation for computer vision applications like sorting systems, interactive installations, or quality control tools.

## Author
Name: Zbigniew Milko            
Gmail: milkozbigniew@gmail.com           
Phone Number: +58 0412-0812321             
