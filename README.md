# Simple CT Viewer ![ct-scan(1)](https://github.com/user-attachments/assets/77cfe3c5-868e-4564-b7e5-eda55689983e)

- making a simple visualization tool for CT volumes | images in many formate [ nrrd , npy , dicom ....etc]

<img width="1390" height="919" alt="Screenshot 2025-09-01 020346" src="https://github.com/user-attachments/assets/f5d31968-065a-4585-9eb7-4675996af6c0" />

<img width="1398" height="922" alt="Screenshot 2025-08-17 111353" src="https://github.com/user-attachments/assets/d4e3dd55-56a8-433e-a6d4-fa20755fd5ea" />


## How to use :
1. [pip install requirements] (recommended to install in virtual env)
2.  run [python main.py]
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
