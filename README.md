# Real-time Object Detection with Object-Orion

This project implements a real-time object detection application using a custom YOLO-based model called Object-Orion. The application is built with Streamlit and offers multiple input options including image upload, video upload, and real-time camera feed.

## Try out the app.

[Object-Orion](object-orion.streamlit.app)

## Features

- Real-time object detection using Object-Orion v1
- Support for image, video, and camera input
- User-friendly interface built with Streamlit
- Asynchronous processing for smooth performance

## Model: Object-Orion

Object-Orion is a custom object detection model based on the YOLO (You Only Look Once) architecture. It's currently in its first version (v1) with an accuracy of approximately 60%. While it's still in development, it demonstrates promising results for real-time object detection tasks.

For more information about Object-Orion, visit [Object-Orion](https://github.com/gauravvvvvvvvvv/object-orion).

## Project Structure

The project consists of two main Python files:

1. `app.py`: The main application file that creates the Streamlit interface and handles user interactions.
2. `yolo_predictions.py`: Contains the `YOLO_Pred` class that interfaces with the Object-Orion model for making predictions.

### app.py

This file sets up the Streamlit application and includes the following key components:

- `YOLOTransformer` class: Handles real-time video processing using the Object-Orion model.
- Input selection: Allows users to choose between image, video, or camera input.
- Image and video processing functions: Handle uploaded images and videos.
- WebRTC implementation: Enables real-time camera feed processing.

### yolo_predictions.py

This file contains the `YOLO_Pred` class, which is responsible for:

- Loading the Object-Orion model
- Processing input images or video frames
- Applying non-maximum suppression to filter detections
- Drawing bounding boxes and labels on the processed images

## Usage

1. Install the required dependencies:
2. Run the Streamlit app:
3. Select your preferred input method (Image, Video, or Camera) from the radio buttons.

4. For image or video input, upload a file using the file uploader.

5. For camera input, allow the application to access your camera when prompted.

6. View the results of object detection in real-time!

## Future Improvements

- Improve the accuracy of the Object-Orion model
- Add support for more object classes
- Optimize performance for faster processing
- Implement object tracking for video and camera inputs

## Contributing

Contributions to improve the application or the Object-Orion model are welcome. Please feel free to submit pull requests or open issues on the GitHub repository.

## License

**MIT LICENSE**
