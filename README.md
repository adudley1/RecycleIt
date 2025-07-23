
# RecycleIt
RecycleIt is a smart waste classification system that identifies whether an object is recyclable, conditionally recyclable, or compostable using image recognition. It runs on an NVIDIA Jetson Orin Nano and leverages deep learning to analyze photos of trash items and categorize them in real time. Users can upload a photo or secretly enter a recycling code to determine disposal instructions.

## The Algorithm
RecycleIt uses a ResNet-18 deep learning model, trained with Jetson Inference tools, to classify images of waste into recyclable, conditionally recyclable, or compostable categories. The Flask-based app accepts uploaded images, runs them through the model on a Jetson Orin Nano, and displays the result. 

## Running this project

## How to Run

1) Download all files, open in Visual Studio Code
2) Open a new terminal window
3) CD to recycle-it (cd recycle-it)
4) Downgrade your numpy (pip install numpy==1.24.4)
5) Run the program (python app.py)
6) When prompted (at the lower right of your screen) press "Open in browser"
7) If you want to quit the program go back to your terminal and press Ctrl+C
8) Click "upload file" and choose an image of a household item you want to classify. It should say the file name next to the button when you import.
9) Press "go" and wait, this normally doesn't take long unless it's your first time using the program.
10) When it's done it should show the image you uploaded as well as what it identifies it to be, it's percent certainty, and if it's recyclable.

VIDEO INSTRUCTIONS: https://github.com/user-attachments/assets/297e09ec-145b-4c78-8c07-9d58e1e58cb1
