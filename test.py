from keras.models import load_model
from PIL import Image, ImageOps
import numpy as np
import cv2 as cv

# Load the model
model = load_model('keras_model.h5')

# Create the array of the right shape to feed into the keras model
# The 'length' or number of images you can put into the array is
# determined by the first position in the shape tuple, in this case 1.

#knn = cv2.ml.KNearest_create()
#knn.train(angle, cv2.ml.ROW_SAMPLE, label)
data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

size = (224, 224)

cap = cv.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open camera")
    exit()
while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    frame = cv.flip(frame, 1)
    # if frame is read correctly ret is True
    if not ret:
        print("카메라를 찾을 수 없습니다 ... 종료합니다")
        break

    # Our operations on the frame come here
    image1 = cv.resize(frame, size, interpolation=cv.INTER_AREA)
    image_array = np.asarray(image1)
    normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1
    data[0] = normalized_image_array
    
       
    prediction = model.predict(data)
    
    #print(prediction)
    if (prediction[0,0] < prediction[0,1]):
        cv.putText(image1, 'Non', (0, 25), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0))
        print("헤드폰 없음")

    else:
        cv.putText(image1, 'Headphone', (0, 25), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255))
        print('헤드폰 있음')
  


    # Display the resulting frameq
    cv.imshow('Webcam', image1)
    if cv.waitKey(1) == ord('q'):
        break
# When everything done, release the capture
cap.release()
cv.destroyAllWindows()

