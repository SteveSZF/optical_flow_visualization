# optical_flow_visualization
效果动图: ![Image text](https://github.com/SteveSZF/optical_flow_visualization/blob/master/test.gif)
## 运行环境
OS:Ubuntu16.04
caffe框架:https://github.com/xmfbit/flownet2.git
python2.7
opencv4.1.0
cuda8.0
cudnn7.0

## 运行
1. 下载FlowNet2_weights.caffemodel.h5模型文件
2. 运行python visual_flow.py --caffemodel ./FlowNet2_weights.caffemodel.h5 --deployproto ./FlowNet2_camera_deploy.prototxt --video /home/szf/Videos/test2.mp4 --gpu 0 --step 4
