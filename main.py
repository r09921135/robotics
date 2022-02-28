from matplotlib import pyplot as plt
import numpy as np
from PIL import Image
import cv2

from speech2text import recordSpeech
from bert.classifier_inference import actionClassify
from bert.inference import splitSpeech
from centroid import findCentroid
from ris import RIS
from args import get_parser


def display(output_mask, image, cX, cY, angle):
    plt.figure()
    plt.axis('off')

    im = np.array(image)
    plt.imshow(im)

    ax = plt.gca()
    ax.set_autoscale_on(False)

    # mask definition
    img = np.ones((im.shape[0], im.shape[1], 3))
    color_mask = np.array([0, 255, 0]) / 255.0
    for i in range(3):
        img[:, :, i] = color_mask[i]

    output_mask = output_mask.transpose(1, 2, 0)
    ax.imshow(np.dstack((img, output_mask * 0.5)))
    ax.axline((cX,cY), slope=np.tan(np.radians(angle)), color='red')
    circle = plt.Circle((cX, cY), color='r')
    ax.add_patch(circle)

    plt.show()
    plt.close()


def main(args):
    ## Record speech command
    speech = recordSpeech()
    # speech = 'can you feed me the bread'
    print("Your command: " + speech)

    ## Split command into action subtext and discription subtext
    action_subtext, object_subtext = splitSpeech(speech)

    ## Classify action 
    action_id = actionClassify(action_subtext)
    action = 'give' if action_id == 0 else 'feed'
    print('Action:', action)

    ## Perform refering image segmentation
    img_path = args.test_img
    image = Image.open(img_path).convert('RGB')
    output_mask = RIS(args, image, object_subtext)

    ## Calculate centroid and angle
    mask = (output_mask.squeeze(0) * 255).astype(np.uint8)
    cX, cY, angle = findCentroid(mask)

    display(output_mask, image, cX, cY, angle)



if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    main(args)