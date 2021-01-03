# obj-mot-dt
Object and motion detection from youtube streaming

### Git clone
```git clone https://github.com/nisaji/obj-mot-dt.git```

### Requirement
```
conda install pandas
conda install pytorch torchvision -c pytorch
pip install cython
conda install opencv
pip install matplotlib
```

### Get weight file
```wget https://pjreddie.com/media/files/yolov3.weights```



### Example usage
```python obj-mot-dt.py "https://www.youtube.com/watch?v=G0IBqtO1K28"```

### Result
![result](https://github.com/nisaji/obj-mot-dt/blob/master/mp4/AGDRec_Tri.mp4?raw=true)
