from enum import Enum

class TaskType(str, Enum):
    """ The types of tasks Florence-2 supports """


    # Captioning - Don't require additional text input
    CAPTION = '<CAPTION>'
    """Image level brief caption"""
    DETAILED_CAPTION = '<DETAILED_CAPTION>'
    """Image level detailed caption"""
    MORE_DETAILED_CAPTION = '<MORE_DETAILED_CAPTION>'
    """Image level very detailed caption"""

    # Object detection - don't require additional text input
    REGION_PROPOSAL = '<REGION_PROPOSAL>'
    """Proposes bounding boxes for salient objects (no labels)"""
    OBJECT_DETECTION = '<OD>'
    """Identifies objects via bounding boxes and gives categorical labels"""
    DENSE_REGION_CAPTION = '<DENSE_REGION_CAPTION>'
    """Identifies objects via bounding boxes and gives natural language labels"""

    PHRASE_GROUNDING = '<CAPTION_TO_PHRASE_GROUNDING>'
    """Given a caption, provides bounding boxes to visually ground phrases in the caption"""
    
    RES = '<REFERRING_EXPRESSION_SEGMENTATION>'
    """Referring Expression Segmentation - given a natural language descriptor
    identifies the segmented region that corresponds
    """

    REG_TO_SEG = '<REGION_TO_SEGMENTATION>'
    """Segments salient object in a given region"""

    OPEN_VOCAB_DETECTION = '<OPEN_VOCABULARY_DETECTION>'
    """Detect bounding box for objects and OCR text"""

    # Region to text
    REGION_TO_CATEGORY = '<REGION_TO_CATEGORY>'
    """ get object classification for bounding box """
    REGION_TO_DESCRIPTION = '<REGION_TO_DESCRIPTION>'
    """ get natural language description for contents of bounding box """

    # OCR
    OCR = '<OCR>'
    """ OCR for entire image """
    OCR_WTIH_REGION = '<OCR_WITH_REGION>'
    """ OCR for entire image, with bounding boxes for individual text items """
    



# prediction function
def run_example(task_prompt: TaskType, model, processor, image, text_input=None):
    # Make sure task is a supported task
    if not isinstance(task_prompt, TaskType):
      raise ValueError(f"""
      task_prompt must be a TaskType, but {task_prompt} is of type {type(task_prompt)})
      """)

    # some prompt types do not require inputs
    if text_input is None:
        prompt = task_prompt
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
        task=task_prompt,
        image_size=(image.width, image.height)
    )

    return parsed_answer