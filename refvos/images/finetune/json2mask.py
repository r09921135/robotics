import json
import numpy as np
import os
import cv2

for i in range(6, 23):
    data_name = f'{i:04}'
    with open(os.path.join('./json', data_name+'.json'), 'r') as f:
        data = f.read()

    data = json.loads(data)

    points = data['shapes'][0]['points']
    points = np.array(points, dtype=np.int32)

    image = cv2.imread(os.path.join('./image', data_name+'.png'))

    mask = np.zeros_like(image[:,:,0], dtype=np.uint8)

    cv2.fillPoly(mask, [points], (255))

    cv2.imwrite(os.path.join('./mask', data_name+'.png'), mask)