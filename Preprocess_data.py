import xml.etree.ElementTree as ET
import numpy as np
import cv2, re
import os
from pathlib import Path
import pandas as pd
from tqdm import tqdm

ink_namespace = {"ink": "http://www.w3.org/2003/InkML"}

def read_inkml(path):
    
    data_tree = ET.parse(str(path)) # makes a tree with all tags as nodes, where the root is the <ink> tag
    root = data_tree.getroot() # gets the root which is the <ink> tag

    # Under the root node, recursively search (using .//), for any tag in the namespace called trace, where ink: is the namespace.
    traces = root.findall(".//ink:trace", ink_namespace) or root.findall(".//trace")

    strokes = []
    for curr_trace in traces:

        text = (curr_trace.text or "").strip() # Extract numbers in trace tag and remove spaces just incase.
        if not text: continue

        points = []
        for p in re.split(r"[,\n]+", text): #Points come in pairs of 3 so split accordingly (refer to the dataset). Other patterns included for safety
            coordinates = re.split(r"[\s,]+", p.strip()) # Split the point to extract each individual coordinate, use strip here incase of any whitespaces

            if len(coordinates) >= 2: #Check against corrupted data
                try:
                    x, y = float(coordinates[0]), float(coordinates[1])
                    points.append([x,y])
                except:
                    pass

        if points:
            strokes.append(np.array(points, dtype=np.float32)) # convert points for that trace into numpy array and add it to strokes
    
    ann = root.find(".//ink:annotation[@type='normalizedLabel']", ink_namespace)

    

    if ann is None:
        ann = root.find(".//ink:annotation[@type='truth']", ink_namespace) \
              or root.find(".//annotation[@type='label']") \
              or root.find(".//annotation[@type='truth']") \
              or root.find(".//ink:annotation[@type='label']", ink_namespace)
        
    
    label = ann.text.strip() if ann is not None and ann.text else ""
    return strokes, label

def rasterize_strokes(strokes, stroke_width = 2, margin = 2, target_h = 128, max_w =768):
    
    if not strokes: # if no strokes, output white box
        return np.ones((target_h,target_h), dtype = np.uint8) * 255
    
    all_points  = np.vstack(strokes) # stack all coordinates into one vector

    min_x, min_y = np.min(all_points, axis = 0) # Search for pair with lowest values ROW-WISE
    max_x, max_y = np.max(all_points, axis = 0) # Search for pair with largest values ROW-WISE

    w, h = int(max_x - min_x) + 2*margin, int(max_y - min_y) + 2*margin # Compute maximum needed width and height and add margin on both sides of each axis

    img = np.ones((h,w), dtype = np.uint8)  * 255 # Set image background to white

    offset = np.array([min_x - margin, min_y - margin], np.float32) # offset needed to push all points

    for s in strokes:

        curr_points  = (s - offset).astype(np.int32) # now min_x and min_y will become "margin" pixels away from left and top borders
        if len(curr_points) == 1:

            cv2.circle(img, tuple(curr_points[0]), stroke_width, 0, -1) #draw a dot

        else:
            cv2.polylines(img, [curr_points], False, 0, stroke_width, lineType=cv2.LINE_AA) # Simulate stroke by connecting all points via anti-aliased lines.


    scale = target_h / img.shape[0] # how much we want to change current image dimensions by
    new_w = int(img.shape[1] * scale) # scale width by that much too to preserve aspect ratio
    img = cv2.resize(img, (new_w, target_h), interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC) # resize image based on if we upscale or downscale

    if new_w < max_w:
        pad = np.ones((target_h, max_w - new_w), np.uint8) * 255 # make pas matrix for extra white background
        img = np.concatenate([img, pad], axis=1) # concat the two matrices
    else:
        img = cv2.resize(img, (max_w, target_h), interpolation=cv2.INTER_AREA) # downscale image if still bigerr than max width


    return img

def preprocess_data(root,out):

    root  = Path(root)
    out = Path(out)

    paths  = sorted(root.rglob("*.inkml"))

    data = [] # will use as dictionary for image name to ground truth mapping

    for p in tqdm(paths, desc = "Converting Inkml to PNG"): # use tqdm for progress bar
        try:

            strokes, label  = read_inkml(p)
            image = rasterize_strokes(strokes)
            
            image_path = out / f"{p.stem}.png" 
            cv2.imwrite(str(image_path), image) #create the actual image to be seen

            data.append({"path": f"{p.stem}.png", "label": f"{label}"}) # create mapping of image to ground truth
        except Exception as e:
             tqdm.write(f"[WARN] {p.name}: {e}") 

    

    pd.DataFrame(data).to_csv(out / "data" / "data.csv", index=False)
    print(f"Done, {len(data)} files processed.")

if __name__ == "__main__": # ensures this isn't run if preprocess is imported anywhere else
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required = True, help = "Path to where data to be preprocessed is.")
    ap.add_argument("--out", required  = True, help = "Path to storing preprocessed data.")
    args = ap.parse_args()
    preprocess_data(args.root, args.out)





    


