#include <iostream>
#include <opencv2/opencv.hpp>
#include <string>
#include <stdio.h>
using namespace std;
using namespace cv;
int main(void){ 
char buff1[100]; char buff2[100]; 
for(int i=1;i<280;i++){ 
sprintf(buff1,"/home/tj816/mask-rcnn/train_data/labelme_json/%d_json/label.png",i); 
sprintf(buff2,"/home/tj816/mask-rcnn/train_data/cv2_mask/%d.png",i);
Mat src;
//Mat dst;
src=imread(buff1,CV_LOAD_IMAGE_UNCHANGED);
Mat ff=Mat::zeros(src.rows,src.cols,CV_8UC1);
for(int k=0;k<src.rows;k++){ 
for(int kk=0;kk<src.cols;kk++){ 
int n=src.at<ushort>(k,kk); 
ff.at<uchar>(k,kk)=n; }
}
imwrite(buff2,ff);
    }
    return 0;
}

