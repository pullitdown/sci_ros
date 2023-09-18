#!/home/sunteng/workspace/conda/envs/py39/bin/python

import rospy
from sensor_msgs.msg import Image
from message_filters import ApproximateTimeSynchronizer, Subscriber


import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import torch
from torchvision import transforms

from gray_model import Finetunemodel

class ImageProcessor:
    def __init__(self):
        # 创建 CvBridge 对象
        self.bridge = CvBridge()
        self.model = "/home/sunteng/catkin_ws/src/EXP/Train-20230915-1103180.5mse1.5smooth/model_epochs/weights_122.pt"
        # 加载你的 PyTorch 模型
        self.device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

        self.model = Finetunemodel(self.model)
        self.model.eval()
        self.model.to(self.device)
        sub_image_0 = rospy.get_param('/sci_node/sub_image_0','/cam0/image_raw')
        sub_image_1 = rospy.get_param('/sci_node/sub_image_1','/cam1/image_raw')
        pub_image_0 = rospy.get_param('/sci_node/pub_image_0','/cam0/image_enhance')
        pub_image_1 = rospy.get_param('/sci_node/pub_image_1','/cam1/image_enhance')
        # 定义图像预处理流程
        self.preprocess = transforms.Compose([
            transforms.ToTensor(),
        ])

        # 创建一个订阅者和发布者
        image1_sub = Subscriber(sub_image_0, Image)
        image2_sub = Subscriber(sub_image_1, Image)

        ats = ApproximateTimeSynchronizer([image1_sub, image2_sub], queue_size=5, slop=0.1)
        ats.registerCallback(self.image_callback)
        self.pub_left = rospy.Publisher(pub_image_0, Image, queue_size=10)
        self.pub_right = rospy.Publisher(pub_image_1, Image, queue_size=10)

    def image_callback(self, msg_0,msg_1):
        # 将 ROS 图像消息转换为 OpenCV 图像
        cv_image_0 = self.bridge.imgmsg_to_cv2(msg_0, "mono8")
        cv_image_1 = self.bridge.imgmsg_to_cv2(msg_1, "mono8")
        input_tensor_0 = self.preprocess(cv_image_0)
        input_tensor_1 = self.preprocess(cv_image_1)
        input_batch = torch.stack([input_tensor_0, input_tensor_1]).to(device=self.device)


        # 将图像输入到模型中
        with torch.no_grad():
            output = self.model(input_batch)

        # 将模型输出转换为图像（这取决于你的模型和需求）
        output_image = self.output_to_image(output)

        # 将 OpenCV 图像转换为 ROS 图像消息
        msg_out_0 = self.bridge.cv2_to_imgmsg(output_image[0], "mono8")
        msg_out_1 = self.bridge.cv2_to_imgmsg(output_image[1], "mono8")

        # 发布处理后的图像
        self.pub_left.publish(msg_out_0)
        self.pub_right.publish(msg_out_1)

    def output_to_image(self, output):
        # 根据你的模型和需求，将模型输出转换为图像
        # 这里是一个简单的例子，将输出张量转换为 8 位的灰度图像
        output_image_0 = (output[1][0].detach().cpu().squeeze().numpy() * 255).astype('uint8')
        output_image_1 = (output[1][1].detach().cpu().squeeze().numpy() * 255).astype('uint8')
        return output_image_0,output_image_1

if __name__ == '__main__':
    rospy.init_node('sci_node')
    image_processor = ImageProcessor()
    rospy.spin()
