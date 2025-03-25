import cv2
from src.utils import *
from src.post_proc_slam import FloorPlanProcessor as SLAM
import argparse
import os

def parse_arguments():
    parser = argparse.ArgumentParser(description='Post-process SLAM output')
    parser.add_argument('--image_dir', type=str, required=True, help='Path to the directory containing the images')
    parser.add_argument('--yaml_path', type=str, required=True, help='Path to the YAML file containing metadata')
    parser.add_argument('--output_dir', type=str, required=True, help='Path to the output directory')
    parser.add_argument('--output_format', type=str, choices=['png', 'jpg'], required=True, help='Output image format (png, jpg)')
    parser.add_argument('--vector_format', type=str, choices=['yes', 'no'], required=True, help='Generate vector format (yes/no)')
    parser.add_argument('--vector_choice', type=str, choices=['svg', 'json', 'dxf'], help='Choice of vector format (svg, json, dxf)')
    return parser.parse_args()


def main():
    
    args = parse_arguments()
    
    # make a list of paths that ends with .pgm in the image directory
    image_paths = [os.path.join(args.image_dir, f) for f in os.listdir(args.image_dir) if f.endswith('.pgm')]
    
    
    images, metadata = load_image_and_metadata(image_paths, args.yaml_path)
    slam = SLAM()
    
    for i, image in enumerate(images):
        # Use YAML information for preprocessing
        binary_image = preprocess_image(
            image,
            metadata['occupied_thresh'],
            metadata['free_thresh'],
            metadata['negate']
        )

        edges = slam.detect_edges(binary_image)
        lines = slam.detect_lines(edges, metadata['resolution'])
        
        floor_plan = slam.draw_floor_plan(lines, image)
        
        if not os.path.exists(args.output_dir): #check for output directory
            os.makedirs(args.output_dir)
            
        # Save the floor plan image
        image_name = os.path.basename(image_paths[i]).split('.')[0]
        output_path = os.path.join(args.output_dir, f"{image_name}.{args.output_format}")
        cv2.imwrite(output_path, floor_plan)
        
        if args.vector_format == 'yes':
            if args.vector_choice == 'svg':
                output_path = os.path.join(args.output_dir, f"{image_name}.svg")
                slam.export_as_svg(lines, output_path)
            elif args.vector_choice == 'json':
                output_path = os.path.join(args.output_dir, f"{image_name}.json")
                slam.export_as_json(lines, output_path)
            elif args.vector_choice == 'dxf':
                output_path = os.path.join(args.output_dir, f"{image_name}.dxf")
                slam.export_as_dxf(lines, output_path)
            else:
                raise ValueError("Invalid vector format choice. Expected one of 'svg', 'json', 'dxf'.")
         

if __name__ == "__main__":
    main()