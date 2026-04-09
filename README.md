In order to run this app some libraries are needed. The yml file provided in the Extras folder titled "torchcuda" can be loaded up in a virtual environment like anaconda in order to run the code without having to worry about downloading all libraries.

If you are choosing to run this application on anaconda, you should:
    1. First cd into the Extras folder and type in the command "conda env create -f torchcuda.yml" and then activate this environment with the command "conda activate torchcuda".
    2. cd out of the extras folder and run the app by typing "python MathEquationSolverApp.py". This will take a few seconds to open up.
    3. After it runs, you will notice the line that indicates the app is running on a local url, specifically "* Running on local URL:  http://127.0.0.1:7860". Crtl+click on this and the website will open.
    4. You should be able to upload a handwritten equation equation into the app.

If you are not running this on conda you should just follow steps 2-4 above while making sure the required libraries are downloaded.

PLEASE NOTE! The application expects a fully cropped, uniform (preferably white) background handwritten equation from an electronic device like a tablet or laptop. If any other type of image is included, the model will fail to read the image properly.

Additionally, 3 images that were tested by me are provided below incase the user wants to test those too.

As checkpoints are big in size, not all checkpoints are included. Only the best ones from the best epochs are included.