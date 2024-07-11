from enum import Enum
import random

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# constants
colormap = ['blue','orange','green','purple','brown','pink','gray','olive','cyan','red',
            'lime','indigo','violet','aqua','magenta','coral','gold','tan','skyblue']

# to be set
model = None
processor = None

def set_model_info(model_, processor_):
    global model, processor
    model = model_
    processor = processor_


class TaskType(str, Enum):
    """ The types of tasks Florence-2 supports """

    ########################## NO ADDITIONAL INPUT ############################
    # Whole image to natural language
    CAPTION = '<CAPTION>'
    """Image level brief caption"""
    DETAILED_CAPTION = '<DETAILED_CAPTION>'
    """Image level detailed caption"""
    MORE_DETAILED_CAPTION = '<MORE_DETAILED_CAPTION>'
    """Image level very detailed caption"""

    # Whole image to text( + region)
    OCR = '<OCR>'
    """ OCR for entire image """
    OCR_WITH_REGION = '<OCR_WITH_REGION>'
    """ OCR for entire image, with bounding boxes for individual text items """

    # Whole image to regions (+ categories or natural language)
    REGION_PROPOSAL = '<REGION_PROPOSAL>'
    """Proposes bounding boxes for salient objects (no labels)"""
    OBJECT_DETECTION = '<OD>'
    """Identifies objects via bounding boxes and gives categorical labels"""
    DENSE_REGION_CAPTION = '<DENSE_REGION_CAPTION>'
    """Identifies objects via bounding boxes and gives natural language labels"""

    ############################# REGION INPUT #################################
    # Region to segment
    REG_TO_SEG = '<REGION_TO_SEGMENTATION>'
    """Segments salient object in a given region"""

    # Region to text
    REGION_TO_CATEGORY = '<REGION_TO_CATEGORY>'
    """ get object classification for bounding box """
    REGION_TO_DESCRIPTION = '<REGION_TO_DESCRIPTION>'
    """ get natural language description for contents of bounding box """

    ######################### NATURAL LANGUAGE INPUT ###########################
    # Natural language to regions (1 to many)
    PHRASE_GROUNDING = '<CAPTION_TO_PHRASE_GROUNDING>'
    """Given a caption, provides bounding boxes to visually ground phrases in the caption"""

    # Natural language to region (1 to 1?)
    OPEN_VOCAB_DETECTION = '<OPEN_VOCABULARY_DETECTION>'
    """Detect bounding box for objects and OCR text"""
    
    # Natural language to segment (1 to 1?)
    RES = '<REFERRING_EXPRESSION_SEGMENTATION>'
    """Referring Expression Segmentation - given a natural language descriptor
    identifies the segmented region that corresponds
    """
    

# prediction function
def run_example(task_prompt: TaskType, image, text_input=None):
    # Make sure task is a supported task
    if not isinstance(task_prompt, TaskType):
      raise ValueError(f"""
      task_prompt must be a TaskType, but {task_prompt} is of type {type(task_prompt)})
      """)

    # some prompt types do not require inputs
    if text_input is None:
        prompt = task_prompt.value
    else:
        prompt = task_prompt.value + text_input

    #
    inputs = processor(text=prompt, images=image, return_tensors="pt")
    generated_ids = model.generate(
      input_ids=inputs["input_ids"].cuda(),
      pixel_values=inputs["pixel_values"].cuda(),
      max_new_tokens=1024,
      early_stopping=False,
      do_sample=False,
      num_beams=3,
    )
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
    parsed_answer = processor.post_process_generation(
        generated_text,
        task=task_prompt.value,
        image_size=(image.width, image.height)
    )

    return parsed_answer


def plot_bbox(data, image):
    """given an image and a dictionary of bounding boxes, 
    plot the bounding boxes on the image"""

   # Create a figure and axes
    fig, ax = plt.subplots()

    # Display the image
    ax.imshow(image)

    # Plot each bounding box
    for bbox, label in zip(data['bboxes'], data['labels']):
        # Unpack the bounding box coordinates
        x1, y1, x2, y2 = bbox
        # Create a Rectangle patch
        rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=1, edgecolor='r', facecolor='none')
        # Add the rectangle to the Axes
        ax.add_patch(rect)
        # Annotate the label
        plt.text(x1, y1, label, color='white', fontsize=8, bbox=dict(facecolor='red', alpha=0.5))

    # Remove the axis ticks and labels
    ax.axis('off')

    # Show the plot
    plt.show()


def draw_polygons(image, prediction, fill_mask=False):
    """
    Draws segmentation masks with polygons on an image.

    Parameters:
    - image_path: Path to the image file.
    - prediction: Dictionary containing 'polygons' and 'labels' keys.
                  'polygons' is a list of lists, each containing vertices of a polygon.
                  'labels' is a list of labels corresponding to each polygon.
    - fill_mask: Boolean indicating whether to fill the polygons with color.
    """
    # Load the image
    draw = ImageDraw.Draw(image)


    # Set up scale factor if needed (use 1 if not scaling)
    scale = 1

    # Iterate over polygons and labels
    for polygons, label in zip(prediction['polygons'], prediction['labels']):
        color = random.choice(colormap)
        fill_color = random.choice(colormap) if fill_mask else None

        for _polygon in polygons:
            _polygon = np.array(_polygon).reshape(-1, 2)
            if len(_polygon) < 3:
                print('Invalid polygon:', _polygon)
                continue

            _polygon = (_polygon * scale).reshape(-1).tolist()

            # Draw the polygon
            if fill_mask:
                draw.polygon(_polygon, outline=color, fill=fill_color)
            else:
                draw.polygon(_polygon, outline=color)

            # Draw the label text
            draw.text((_polygon[0] + 8, _polygon[1] + 2), label, fill=color)

    # Save or display the image
    #image.show()  # Display the image
    #display(image)
    return image


def convert_to_od_format(data):
    """
    Converts a dictionary with 'bboxes' and 'bboxes_labels' into a dictionary with separate 'bboxes' and 'labels' keys.

    Parameters:
    - data: The input dictionary with 'bboxes', 'bboxes_labels', 'polygons', and 'polygons_labels' keys.

    Returns:
    - A dictionary with 'bboxes' and 'labels' keys formatted for object detection results.
    """
    # Extract bounding boxes and labels
    bboxes = data.get('bboxes', [])
    labels = data.get('bboxes_labels', [])

    # Construct the output format
    od_results = {
        'bboxes': bboxes,
        'labels': labels
    }

    return od_results

def draw_ocr_bboxes(image, prediction):
    scale = 1
    draw = ImageDraw.Draw(image)
    bboxes, labels = prediction['quad_boxes'], prediction['labels']
    for box, label in zip(bboxes, labels):
        color = random.choice(colormap)
        new_box = (np.array(box) * scale).tolist()
        draw.polygon(new_box, width=3, outline=color)
        draw.text((new_box[0]+8, new_box[1]+2),
                    "{}".format(label),
                    align="right",

                    fill=color)
    return image


def convert_bbox_to_relative(box, image):
  """ converts bounding box pixel coordinates to relative coordinates in the
  range 0-999 """
  return [
      (box[0]/image.width)*999,
      (box[1]/image.height)*999,
      (box[2]/image.width)*999,
      (box[3]/image.height)*999,
    ]

def convert_relative_to_bbox(relative, image):
  """ converts list of relative coordinates to pixel coordinates """
  return [
      (relative[0]/999)*image.width,
      (relative[1]/999)*image.height,
      (relative[2]/999)*image.width,
      (relative[3]/999)*image.height,
    ]

def convert_relative_to_loc(relative_coordinates):
  """ converts a list of relative coordinate positions x1, y1, x2, y2 to a 
  string of position tokens """
  return ''.join([f'<loc_{i}>' for i in relative_coordinates])

def convert_bbox_to_loc(box, image):
  """ convert bounding box pixel coordinates to position tokens """
  relative_coordinates = convert_bbox_to_relative(box, image)
  return convert_relative_to_loc(relative_coordinates)

