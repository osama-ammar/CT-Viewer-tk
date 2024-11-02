# Simple CT Viewer ![ct-scan(1)](https://github.com/user-attachments/assets/77cfe3c5-868e-4564-b7e5-eda55689983e)

- making a simple visualization tool for CT volumes | images in many formate [ npy , dicom ....etc]
- [download] (https://drive.google.com/file/d/1_6TlsJRiXuZEukuEZeoLVZsdoC0f6kHG/view?usp=drive_link)


![Captureaas](https://github.com/user-attachments/assets/57ab0a3f-fd13-4132-8f2e-76a23088e962)

![Capture](https://github.com/user-attachments/assets/7cf5391f-a38b-48e5-81dd-afe5bb659e88)

## How to use :
1. [pip install requirements] (recommended to install in virtual env)
2.  run [python viewer.py]
3. (optional) if you wanna get executible [pyinstaller --onefile --icon=icon.ico viewer.py]


### TODOS
- [\] accepting diom , npy , NIFTI , others ,images
- [x] fix loading error
- [ ] making a batch for generating the executible
- [ ] updating ui in new branch
- [x] opening the app from command line with images path to open
- [x] loading 2D image option
- [ ] inserting a segmentation|detection model
    - [ ] use simple model to detect|segment selected image
    - [ ] show the segmentation on new window
    - [ ] save the segmentation
