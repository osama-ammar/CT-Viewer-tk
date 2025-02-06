# Simple CT Viewer ![ct-scan(1)](https://github.com/user-attachments/assets/77cfe3c5-868e-4564-b7e5-eda55689983e)

- making a simple visualization tool for CT volumes | images in many formate [ npy , dicom ....etc]
- [download](https://drive.google.com/file/d/1gCQI09gq5nAa0AqI_JqfL9vdT35mjNGE/view?usp=sharing)


![Captureaas](https://github.com/user-attachments/assets/57ab0a3f-fd13-4132-8f2e-76a23088e962)

![Capture](https://github.com/user-attachments/assets/7cf5391f-a38b-48e5-81dd-afe5bb659e88)

## How to use :
1. [pip install requirements] (recommended to install in virtual env)
2.  run [python viewer.py]
3. (optional) if you wanna get executible [pyinstaller --onefile --icon=icon.ico viewer.py]


### TODOS
- [x] accepting dicom , npy and  nrrd.
- [x] fix loading error
- [x] making a batch for generating the executible
- [ ] get all fuctionalities as small functions
- [ ] handle possible exceptions
    - [ ] open 2d image with 3d button and vice verse
    - [ ] open nrrd image with 3d button and vice verse
    - [ ] it's better to make one button that open any format (the app will detect the format)
- [x] opening the app from command line with images path to open
- [x] loading 2D image option
- [ ] inserting a segmentation|detection model
    - [ ] use simple model to detect|segment selected image
    - [ ] show the segmentation on volume|new window
    - [ ] save the segmentation
- [ ] export images as :
    - [ ] npy
    - [ ] nrrd
    - [ ] NIfTI 
    - [ ] TIFF 
- [ ] converting app into web based , utilizing current functionalities .....( Later)