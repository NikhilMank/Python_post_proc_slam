import cv2
import numpy as np
import svgwrite
import json
import ezdxf


class FloorPlanProcessor:
    def __init__(self):
        pass

    def detect_edges(self, binary_image, low_threshold=100, high_threshold=200):
        """Detects the edges in the images using the Canny edge detector.

        Args:
            binary_image (numpy.ndarray): image for edge detection
            low_threshold (int): lower threshold for edge detection
            high_threshold (int): higher threshold for edge detection

        Returns:
            numpy.ndarray: image with edges detected
        """
        return cv2.Canny(binary_image, low_threshold, high_threshold)

    def detect_lines(self, edges, rho=1.7, theta=np.pi/900, threshold=40, min_line_length=40, max_line_gap=35):
        """Detect lines in the image using the Hough Line Transform.

        Args:
            edges (numpy.ndarray): Image with edges detected
            rho (float, optional): Distance resolution for pixel accumulation. Defaults to 1.7.
            theta (float, optional): Angle resolution for accumulation in radians. Defaults to np.pi/900.
            threshold (int, optional): Accumulator threshold. Defaults to 40.
            min_line_length (int, optional): The minimum length of a line segment to be considered valid. Defaults to 40.
            max_line_gap (int, optional): The maximum allowed gap between points on the same line to link them into a single line segment. Defaults to 35.

        Returns:
            numpy.ndarray: array of detected lines, each line in the format [[[x1, y1, x2, y2]]]
        """
        return cv2.HoughLinesP(edges, rho, theta, threshold, minLineLength=min_line_length, maxLineGap=max_line_gap)

    def draw_floor_plan(self, lines, image):
        """Draws the detected lines on a blank image to create a floor plan.

        Args:
            lines (numpy.ndarray): array of detected lines in the format [[[x1, y1, x2, y2]]]
            image (numpy.ndarray): image to create the blank image with the same dimensions

        Returns:
            numpy.ndarray: image with the floor plan drawn
        """
        floor_plan = np.zeros_like(image, dtype=np.uint8)
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                cv2.line(floor_plan, (x1, y1), (x2, y2), 255, 2)
        return floor_plan

    def export_as_svg(self, lines, filename):
        """Exports the detected lines as an SVG file.

        Args:
            lines (numpy.ndarray): array of detected lines in the format [[[x1, y1, x2, y2]]]
            filename (str): name of the SVG file including the path
        Returns:
            None
        """
        
        lines_ = lines[:, 0, :]  # Extract the coordinates from the array
        # Determine the bounding box
        x_min = np.min(lines_[:, [0, 2]])  # Minimum x
        x_max = np.max(lines_[:, [0, 2]])  # Maximum x
        y_min = np.min(lines_[:, [1, 3]])  # Minimum y
        y_max = np.max(lines_[:, [1, 3]])  # Maximum y

        # Create the SVG drawing with a viewbox
        dwg = svgwrite.Drawing(filename, size=(float(x_max - x_min), float(y_max - y_min)), profile='tiny')
        dwg.viewbox(minx=x_min, miny=y_min, width=x_max - x_min, height=y_max - y_min)
        for line in lines:
            x1, y1, x2, y2 = line[0]
            dwg.add(dwg.line((float(x1), float(y1)), (float(x2), float(y2)), stroke=svgwrite.rgb(0, 0, 0, '%')))
        dwg.save()

    def export_as_json(self, lines, filename):
        """ Exports the detected lines as a JSON file.

        Args:
            lines (numpy.ndarray): array of detected lines in the format [[[x1, y1, x2, y2]]]
            filename (str): name of the JSON file including the path
        Returns:
            None
        """
        lines = lines[:, 0, :]  # Extract the coordinates

        # Convert lines to a list of dictionaries
        line_data = [{'start': (int(x1.item()), int(y1.item())), 'end': (int(x2.item()), int(y2.item()))} 
                    for x1, y1, x2, y2 in lines]

        with open(filename, 'w') as f:
            json.dump(line_data, f, indent=4)

    def export_as_dxf(self, lines, filename):
        """ Exports the detected lines as a DXF file.

        Args:
            lines (numpy.ndarray): array of detected lines in the format [[[x1, y1, x2, y2]]]
            filename (str): name of the DXF file including the path
        """
        doc = ezdxf.new(dxfversion='R2010')
        msp = doc.modelspace()
        for line in lines:
            x1, y1, x2, y2 = line[0]
            msp.add_line((x1, y1), (x2, y2))
        doc.saveas(filename)