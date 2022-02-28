#!/usr/bin/env python
import rclpy
import cv2
import numpy as np
import sys
import matplotlib.pyplot as plt
from PIL import Image
import time
from tm_msgs.msg import *
from tm_msgs.srv import *

import pyrealsense2 as rs
import speech_recognition as sr
import face_recognition

from speech2text import recordSpeech
from bert.inference import splitSpeech
from bert.classifier_inference import actionClassify
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
    # ax.axline((cX,cY), slope=np.tan(np.radians(angle)), color='blue')
    circle = plt.Circle((cX, cY), color='r')
    ax.add_patch(circle)

    plt.show()
    plt.close()

# arm client
def send_script(script):
    arm_node = rclpy.create_node('arm')
    arm_cli = arm_node.create_client(SendScript, 'send_script')

    while not arm_cli.wait_for_service(timeout_sec=1.0):
        arm_node.get_logger().info('service not availabe, waiting again...')

    move_cmd = SendScript.Request()
    move_cmd.script = script
    arm_cli.call_async(move_cmd)
    arm_node.destroy_node()

# gripper client
def set_io(state):
    gripper_node = rclpy.create_node('gripper')
    gripper_cli = gripper_node.create_client(SetIO, 'set_io')

    while not gripper_cli.wait_for_service(timeout_sec=1.0):
        node.get_logger().info('service not availabe, waiting again...')
    
    io_cmd = SetIO.Request()
    io_cmd.module = 1
    io_cmd.type = 1
    io_cmd.pin = 0
    io_cmd.state = state
    gripper_cli.call_async(io_cmd)
    gripper_node.destroy_node()

def main(args=None):
    rclpy.init(args=args)
    while True:
        ##initial position
        set_io(0.0)  ##open 
        targetP1 = "200.00, 200, 600, -180.00, 0.0, 135.00"  
        script = "PTP(\"CPP\","+targetP1+",100,200,0,false)"
        send_script(script)
        ##speech recognition
        parser = get_parser()
        args = parser.parse_args()
        speech = recordSpeech()
        print("Your command: " + speech)
        part_action, part_object = splitSpeech(speech)
        
        ##bert動作分類 give/feed
        action_id = actionClassify(part_action)[0]
        
        ##realsense拍照部分
        pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30) 
        pipeline.start(config) 
        sensor = pipeline.get_active_profile().get_device().query_sensors()[1]
        sensor.set_option(rs.option.exposure, 800)
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        color_image = np.asanyarray(color_frame.get_data())
        color_image = Image.fromarray(cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB))  ##from cv image to pil image 
        pipeline.stop()
        ##Ris model
        output_mask = RIS(args, color_image, part_object)
        mask = (output_mask.squeeze(0) * 255).astype(np.uint8)
        cX, cY, angle = findCentroid(mask)
        display(output_mask, color_image, cX, cY, angle)
        
        ##轉換image centroid 到robot coordinate
        r=0.85
        cMat=np.array([cX*r,cY*r,1])
        T_cm = np.array([[0.72731, -0.76301, 238.4358], 
                        [-0.6827, -0.72732, 668.2166], 
                        [0, 0, 1]])
        Rob = np.matmul(T_cm, cMat)
        targetP1 = "%f , %f , 155.00 , 180.00 , 0.00 , %.2f "%(Rob[0], Rob[1], -(angle)+45)   ##夾取target command
        script = "PTP(\"CPP\","+targetP1+",100,200,0,false)"
        send_script(script)
        time.sleep(10)
        set_io(1.0)     # close gripper
        ##tracking part
        targetP1 = "300.00, 300, 650, -270.00, 00.0, 135.00"   ##將手臂移動到待追蹤位置
        script = "PTP(\"CPP\","+targetP1+",100,200,0,false)"
        send_script(script)
        time.sleep(5)   ##停止5秒確保手臂移動到位置了
        print('tracking start')  
        pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        pipeline.start(config)
        arm_to_camera = np.array([[0.707,-0.707,0],[0,0,-1], [0.707,0.707,0]])
        camera_to_arm = np.linalg.inv(arm_to_camera)
        frames = pipeline.wait_for_frames()      
        i=0
        try:
            while True:    ##visual tracking while loop
                ##realsense啟動
                frames = pipeline.wait_for_frames()             
                depth_frame = frames.get_depth_frame()
                color_frame = frames.get_color_frame()
                if not depth_frame or not color_frame:
                    continue
                depth_image = np.asanyarray(depth_frame.get_data())
                color_image = np.asanyarray(color_frame.get_data())
                depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
                face_landmarks_list = face_recognition.face_landmarks(color_image)
                points = []
                ##嘴部偵測
                if len(face_landmarks_list) == 0:    
                    cv2.putText(color_image, 'No face detect!', (10, 40), cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), 1, cv2.LINE_AA)
                else:
                    face_landmark = face_landmarks_list[0]
                    cv2.putText(color_image, 'face detect!', (10, 40), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 1, cv2.LINE_AA)
                    for point in face_landmark['top_lip']:
                        cv2.circle(color_image, point, 1, (255, 0, 0), 4)
                        points.append(point)
                    for point in face_landmark['bottom_lip']:
                        cv2.circle(color_image, point, 1, (255, 0, 0), 4)
                        points.append(point)
                x_axis = []
                y_axis = []
                for (x, y) in points:
                    x_axis.append(x)
                    y_axis.append(y)
                if(len(x_axis) != 0):    ##若有偵測到嘴巴則繪製輪廓
                    center_point = (int(sum(x_axis)/len(x_axis)), int(sum(y_axis)/len(y_axis)))
                    cv2.circle(color_image, center_point, 2, (0, 255, 0), 4)
                    text_depth = "depth value of point center is "+str(np.round(depth_frame.get_distance(center_point[0],center_point[1]),4))+"meter(s)"
                    color_image = cv2.circle(color_image,(center_point[0],center_point[1]),1,(0,255,255),-1)
                    color_image=cv2.putText(color_image, text_depth, (10,20),  cv2.FONT_HERSHEY_PLAIN, 1, (0,0,255), 1, cv2.LINE_AA)
                    print(center_point,depth_frame.get_distance(center_point[0],center_point[1]))
                    z=depth_frame.get_distance(center_point[0],center_point[1])
                    x_dis = 0
                    y_dis = 0
                    if center_point[0] - 320 > 40:    ##若嘴部X軸與中心差距大於+40 x位移10
                        x_dis = 10
                    if center_point[0] - 320 < -40:   ##若嘴部X軸與中心差距大於-40 x位移-10
                        x_dis = -10
                    if center_point[1] - 240 > 40:    ##若嘴部Y軸與中心差距大於+40 y位移10
                        y_dis = 10
                    if center_point[1] - 240 < -40:   ##若嘴部Y軸與中心差距大於-40 y位移-10
                        y_dis = -10
                    if x_dis==0 and y_dis==0 and z != 0:    ##若x.y位移為0且depth 不為0 計數+1
                        i+=1
                    else:
                        i=0
                    if i>4:                         ##若4次迴圈人臉皆沒動(記數>5)則跳出迴圈
                        cv2.destroyAllWindows()
                        break
                    displacement = np.array([x_dis, y_dis, 0])
                    arm_displacement = np.dot(camera_to_arm, displacement)
                    targetP1 = "%f, %f, %f, 0, 0, 0" %(arm_displacement[0], arm_displacement[1], arm_displacement[2])    ##相對移動x位移與y位移
                    script = "Move_PTP(\"CPP\","+targetP1+",100,200,0,false)"
                    send_script(script)
                    time.sleep(0.4)

                images = np.hstack((color_image, depth_colormap))
                cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
                cv2.imshow('RealSense', images)
                key = cv2.waitKey(1)
                if key & 0xFF == ord('q') or key == 27:    
                    cv2.destroyAllWindows()
                    break
        finally:
            pipeline.stop()
        if z>0.3 and action_id==0:               ## 往前到手附近(give me part)
            targetP1 = "0, 0, -300, 0, 0, 0" 
            script = "Move_PTP(\"CPP\","+targetP1+",100,200,0,false)"
            send_script(script)
            time.sleep(3)
            z=z-0.25
            z=z*1000*0.707
            targetP1 = "%f, %f, 0, 0, 0, 0" %(z, z)
            script = "Move_PTP(\"CPP\","+targetP1+",100,200,0,false)"
            send_script(script)
            time.sleep(3)
        if z>0.3 and action_id==1:               ##往前到嘴巴附近 (feed me part)
            z=z-0.15
            z=z*1000*0.707
            targetP1 = "%f, %f, 0, 0, 0, 0" %(z, z)
            script = "Move_PTP(\"CPP\","+targetP1+",100,200,0,false)"
            send_script(script)
            time.sleep(3)
            targetP1 = "0, 0, 80, 0, 0, 0" 
            script = "Move_PTP(\"CPP\","+targetP1+",100,200,0,false)"
            send_script(script)
            time.sleep(3)

        speech2 = recordSpeech()
        print("Your command: " + speech2)
        if "open" in speech2   :    ##若接收音訊有open 則進行動作
            set_io(0.0)
            time.sleep(4)
            targetP1 = "200.00, 200, 600, -180.00, 0.0, 135.00"  ##到初始位置
            script = "PTP(\"CPP\","+targetP1+",100,200,0,false)"
            send_script(script)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
    

