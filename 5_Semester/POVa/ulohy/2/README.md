## 2nd Home Assignemt - Mouse detection in natural environment. 

In this second assignment, you will learn about object detection. You will train your own YOLO object detector
and use it to track objects in a video.

---

You will have to:
1. Process the provided annotations into suitable format for training a YOLO detector.

    1.1. You will also have to create YAML file with paths to the data, which you pass to the CLI train command.

2. Train your own YOLO nano detector using the Ultralytics package. You'll have to choose good training parameters.

3. Complete the infer.py script and use it to track objects in the provided video with your trained model.

---

This assignment consists of (download from E-Learning):
1. data/

    1.1. train/ -- Training data

    1.2. val/ -- Validation data

    1.3. video.mp4 -- Video for testing

    1.4. annotation.json -- Annotations

    1.5. example.yaml -- Example dataset YAML file. Update it to match 
        your local paths and dataset classes.
2. infer.py -- Template for video tracking script.

---

Format of annotation.json:
- Contains a list, where each item corresponds to one image.

- Each item in the lists consists of multiple annotated objects under the "label" key

- CAREFUL: Bounding box positions and sizes are percentages of the image size (0 - 100)

- CAREFUL: 'x' and 'y' are top left corner of a bounding box

---

Single object annotation example:
```
{  
    "x": 24.32851239669421,   # bounding box left position
    "y": 13.223140495867769,  # bounding box top position
    "width": 7.515495867768596,  # bounding box width
    "height": 6.887052341597796, # bounding box height
    "rotation": 0,   # ignore this
    "rectanglelabels": [    # the class of the object - always single class
        "mouse"
    ],
    "original_width": 1024,    # image width - you can ignore this
    "original_height": 576     # image height - you can ignore this
},
```

---

SUBMIT infer.py, trained NANO model (file best.pth), all training output (*.png, *.jpg, results.csv, args.yaml).

Pack all with zip.

MAKE SURE YOU ARE TRAINING THE NANO VARIANT OF THE YOLO DETECTOR

---

My recommendations:
- Use the CLI training instead of writing a python script (see link below)

- pass the 'project=/path' option to the train script, it will save the results to the specified folder

- CAREFUL: When you run the training, the API will download a model.pt file, this is only the pretrained YOLO model, the trained model will be saved in the output directory as weights/(last|best).pt
- Experiment with training options: image size, epochs, learning rate, augmentations, dropout.
- In 800x800 resolution, training requires  roughly 4G of GPU memory.
- Using augmentation and dropout might prevent overfitting (pass augment=True and dropout=0.X) to the train command
- You can start by training for a few epochs (<10), but my model did not detect anything on the validation images. I got acceptable results at around 75+ epochs. Feel free to experiment.
- When creating this assignment, I used the YOLOv8n, you can experiment with newer YOLO versions, for examle the yolo11n.

---

Resources:
- JSON processing: https://docs.python.org/3/library/json.html
- du1 -- For video processing
- Ultralytics documentation:
    - https://docs.ultralytics.com/modes/train/
    - https://docs.ultralytics.com/datasets/detect/#ultralytics-yolo-format
    - https://docs.ultralytics.com/yolov5/tutorials/train_custom_data/

---

You need:
- python3
- cv2
- ultralytics

Send any questions to Martin Kostelník (ikostelnik@fit.vut.cz or Discord) and Michal Hradiš (ihradis@fit.vut.cz).
