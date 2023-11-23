/**
* This file is part of ORB-SLAM2.
* This file is based on the file orb.cpp from the OpenCV library (see BSD license below).
*
* Copyright (C) 2014-2016 Raúl Mur-Artal <raulmur at unizar dot es> (University of Zaragoza)
* For more information see <https://github.com/raulmur/ORB_SLAM2>
*
* ORB-SLAM2 is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* ORB-SLAM2 is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with ORB-SLAM2. If not, see <http://www.gnu.org/licenses/>.
*/
/**
* Software License Agreement (BSD License)
*
*  Copyright (c) 2009, Willow Garage, Inc.
*  All rights reserved.
*
*  Redistribution and use in source and binary forms, with or without
*  modification, are permitted provided that the following conditions
*  are met:
*
*   * Redistributions of source code must retain the above copyright
*     notice, this list of conditions and the following disclaimer.
*   * Redistributions in binary form must reproduce the above
*     copyright notice, this list of conditions and the following
*     disclaimer in the documentation and/or other materials provided
*     with the distribution.
*   * Neither the name of the Willow Garage nor the names of its
*     contributors may be used to endorse or promote products derived
*     from this software without specific prior written permission.
*
*  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
*  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
*  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
*  FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
*  COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
*  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
*  BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
*  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
*  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
*  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
*  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
*  POSSIBILITY OF SUCH DAMAGE.
*
*/


#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <vector>

#include "ORBextractor.h"


using namespace cv;
using namespace std;

namespace ORB_SLAM2
{

const int PATCH_SIZE = 31;
const int HALF_PATCH_SIZE = 15;
const int EDGE_THRESHOLD = 19;

/**
 * 计算特征点的方向,实现特征不变性
 * @param1  特征点所在的图像
 * @param2  特征点在这个图层中的坐标
 * @param3  每个特征点所在的图像区块的每行的边界 vector容器
 * ?原理图：P16
*/
static float IC_Angle(const Mat& image, Point2f pt,  const vector<int> & u_max)
{
    //图像的矩，前者是按照图像块的y坐标加权，后者是按照图像块的x坐标加权
    int m_01 = 0, m_10 = 0;

    //获得这个特征点所在的图像块的中心点坐标的灰度值的指针
    const uchar* center = &image.at<uchar> (cvRound(pt.y), cvRound(pt.x));

    // Treat the center line differently, v=0
    //这条v=0的中心线的计算需要特殊对待
    //由于是中心行+若干行对 所以PATCH_SIZE应该是个奇数
    for (int u = -HALF_PATCH_SIZE; u <= HALF_PATCH_SIZE; ++u)
        m_10 += u * center[u];

    // Go line by line in the circuI853lar patch
    //step1()表示图像一行中包含的字节总数
    int step = (int)image.step1();

    //这里以v=0中心线为对称轴，然后对每一对对称的行进行遍历，加快了计算速度////没必要，可读性差
    for (int v = 1; v <= HALF_PATCH_SIZE; ++v)
    {
        // 
        int v_sum = 0;
        //以获取某行像素横坐标的最大范围
        int d = u_max[v];
        /**
         * 在坐标范围内挨个像素遍历，但实际一次遍历两个
         * 假设每次处理的两个点坐标，中心线上方为(x,y)，中心线下方(x,-y)
         * 对于某次带处理的两个点：m_10= (求和)x*I(x,y) = x*I(x,y)+x*I(x,-y) = x*(I(x,y)+I(x,-y))
         * 对于某次带处理的两个点：m_01= (求和)y*I(x,y) = y*I(x,y)-y*I(x,-y) = y*(I(x,y)-I(x,-y))
        */
        for (int u = -d; u <= d; ++u)
        {   
            //得到需要进行加运算和减运算的像素灰度值
            //val_plus：在中心线下方x=u时的像素灰度值
            //val_minus：在中心线上方x=u时的像素灰度值
            int val_plus = center[u + v*step], val_minus = center[u - v*step];

            //这层循环对u即x遍历，而v即y不变，所以计算每一对点的I(x,y)-I(x,-y)并累加求和，在循环外再乘以v
            v_sum += (val_plus - val_minus);
            //累加计算m_10
            m_10 += u * (val_plus + val_minus);
        }
        m_01 += v * v_sum;
    }

    //为了加快速度还使用了fastAtan2函数，返回特征点的方向角度【0-360】，精度为0.3度
    return fastAtan2((float)m_01, (float)m_10);
}



//乘数因子 一度对应多少弧度
const float factorPI = (float)(CV_PI/180.f);
//计算某个特征点的描述子
static void computeOrbDescriptor(const KeyPoint& kpt,   //要计算描述子的关键点
                                 const Mat& img,        //关键点所在的图像
                                 const Point* pattern,  //随机点集的首地址
                                 uchar* desc            //提取出来的描述子的保存位置
                                )
{   
    //得到特征点的角度，用弧度制表示。kpt.angle是角度制，范围为[0, 360]度
    float angle = (float)kpt.angle*factorPI;

    //计算角度的余弦值和正弦值
    float a = (float)cos(angle), b = (float)sin(angle);

    //获得图像中心指针
    const uchar* center = &img.at<uchar>(cvRound(kpt.pt.y), cvRound(kpt.pt.x));

    //获取图像每行的字节数
    const int step = (int)img.step;

    /**
     * 计算具有旋转不变性的描述子，称之为：Steer BRIEF
     * 在计算的时候需要将这里选取的随机点点集的x轴方向旋转到特征点的方向
    */
    //获得随机“相对点集”中某个idx所对应的点的灰度，这里旋转前坐标为(x,y),旋转后坐标为(x',y')
    //?推导过程P15 x'=xcos(0)-ysin(0)  y'=xsin(0)+ycos(0)
    #define GET_VALUE(idx) \
        center[cvRound(pattern[idx].x*b + pattern[idx].y*a)*step + \
               cvRound(pattern[idx].x*a - pattern[idx].y*b)]

    //brief描述子由32*8位组成
    //其中每一位是来自于两个像素点灰度的直接比较，每次循环比较8对点即得出8bit结果
    //需要16个点，这也是 pattern+=16 的原因
    for (int i = 0; i < 32; ++i, pattern += 16)
    {
        int t0, t1, val;
        t0 = GET_VALUE(0); t1 = GET_VALUE(1);
        val = t0 < t1;
        t0 = GET_VALUE(2); t1 = GET_VALUE(3);
        val |= (t0 < t1) << 1;
        t0 = GET_VALUE(4); t1 = GET_VALUE(5);
        val |= (t0 < t1) << 2;
        t0 = GET_VALUE(6); t1 = GET_VALUE(7);
        val |= (t0 < t1) << 3;
        t0 = GET_VALUE(8); t1 = GET_VALUE(9);
        val |= (t0 < t1) << 4;
        t0 = GET_VALUE(10); t1 = GET_VALUE(11);
        val |= (t0 < t1) << 5;
        t0 = GET_VALUE(12); t1 = GET_VALUE(13);
        val |= (t0 < t1) << 6;
        t0 = GET_VALUE(14); t1 = GET_VALUE(15);
        val |= (t0 < t1) << 7;

        //保留一个字节的描述子，循环32次，最终desc会存满256bit的描述子
        desc[i] = (uchar)val;
    }//该循环得出该关键点的描述子，256bit 32*8

    //消除宏定义
    #undef GET_VALUE
}


/**
 * 下面是预先定义好的随即点集
 * 256是指可以提取出256bit的描述子信息，每个bit由一对点比较而来
 * 4=2*2，前面的2表示需要两个点进行比较，后面的2是一个点有两个坐标
*/
static int bit_pattern_31_[256*4] =
{
    8,-3, 9,    5/*mean (0), correlation (0)*/,
    4,2, 7,-12/*mean (1.12461e-05), correlation (0.0437584)*/,
    -11,9, -8,2/*mean (3.37382e-05), correlation (0.0617409)*/,
    7,-12, 12,-13/*mean (5.62303e-05), correlation (0.0636977)*/,
    2,-13, 2,12/*mean (0.000134953), correlation (0.085099)*/,
    1,-7, 1,6/*mean (0.000528565), correlation (0.0857175)*/,
    -2,-10, -2,-4/*mean (0.0188821), correlation (0.0985774)*/,
    -13,-13, -11,-8/*mean (0.0363135), correlation (0.0899616)*/,
    -13,-3, -12,-9/*mean (0.121806), correlation (0.099849)*/,
    10,4, 11,9/*mean (0.122065), correlation (0.093285)*/,
    -13,-8, -8,-9/*mean (0.162787), correlation (0.0942748)*/,
    -11,7, -9,12/*mean (0.21561), correlation (0.0974438)*/,
    7,7, 12,6/*mean (0.160583), correlation (0.130064)*/,
    -4,-5, -3,0/*mean (0.228171), correlation (0.132998)*/,
    -13,2, -12,-3/*mean (0.00997526), correlation (0.145926)*/,
    -9,0, -7,5/*mean (0.198234), correlation (0.143636)*/,
    12,-6, 12,-1/*mean (0.0676226), correlation (0.16689)*/,
    -3,6, -2,12/*mean (0.166847), correlation (0.171682)*/,
    -6,-13, -4,-8/*mean (0.101215), correlation (0.179716)*/,
    11,-13, 12,-8/*mean (0.200641), correlation (0.192279)*/,
    4,7, 5,1/*mean (0.205106), correlation (0.186848)*/,
    5,-3, 10,-3/*mean (0.234908), correlation (0.192319)*/,
    3,-7, 6,12/*mean (0.0709964), correlation (0.210872)*/,
    -8,-7, -6,-2/*mean (0.0939834), correlation (0.212589)*/,
    -2,11, -1,-10/*mean (0.127778), correlation (0.20866)*/,
    -13,12, -8,10/*mean (0.14783), correlation (0.206356)*/,
    -7,3, -5,-3/*mean (0.182141), correlation (0.198942)*/,
    -4,2, -3,7/*mean (0.188237), correlation (0.21384)*/,
    -10,-12, -6,11/*mean (0.14865), correlation (0.23571)*/,
    5,-12, 6,-7/*mean (0.222312), correlation (0.23324)*/,
    5,-6, 7,-1/*mean (0.229082), correlation (0.23389)*/,
    1,0, 4,-5/*mean (0.241577), correlation (0.215286)*/,
    9,11, 11,-13/*mean (0.00338507), correlation (0.251373)*/,
    4,7, 4,12/*mean (0.131005), correlation (0.257622)*/,
    2,-1, 4,4/*mean (0.152755), correlation (0.255205)*/,
    -4,-12, -2,7/*mean (0.182771), correlation (0.244867)*/,
    -8,-5, -7,-10/*mean (0.186898), correlation (0.23901)*/,
    4,11, 9,12/*mean (0.226226), correlation (0.258255)*/,
    0,-8, 1,-13/*mean (0.0897886), correlation (0.274827)*/,
    -13,-2, -8,2/*mean (0.148774), correlation (0.28065)*/,
    -3,-2, -2,3/*mean (0.153048), correlation (0.283063)*/,
    -6,9, -4,-9/*mean (0.169523), correlation (0.278248)*/,
    8,12, 10,7/*mean (0.225337), correlation (0.282851)*/,
    0,9, 1,3/*mean (0.226687), correlation (0.278734)*/,
    7,-5, 11,-10/*mean (0.00693882), correlation (0.305161)*/,
    -13,-6, -11,0/*mean (0.0227283), correlation (0.300181)*/,
    10,7, 12,1/*mean (0.125517), correlation (0.31089)*/,
    -6,-3, -6,12/*mean (0.131748), correlation (0.312779)*/,
    10,-9, 12,-4/*mean (0.144827), correlation (0.292797)*/,
    -13,8, -8,-12/*mean (0.149202), correlation (0.308918)*/,
    -13,0, -8,-4/*mean (0.160909), correlation (0.310013)*/,
    3,3, 7,8/*mean (0.177755), correlation (0.309394)*/,
    5,7, 10,-7/*mean (0.212337), correlation (0.310315)*/,
    -1,7, 1,-12/*mean (0.214429), correlation (0.311933)*/,
    3,-10, 5,6/*mean (0.235807), correlation (0.313104)*/,
    2,-4, 3,-10/*mean (0.00494827), correlation (0.344948)*/,
    -13,0, -13,5/*mean (0.0549145), correlation (0.344675)*/,
    -13,-7, -12,12/*mean (0.103385), correlation (0.342715)*/,
    -13,3, -11,8/*mean (0.134222), correlation (0.322922)*/,
    -7,12, -4,7/*mean (0.153284), correlation (0.337061)*/,
    6,-10, 12,8/*mean (0.154881), correlation (0.329257)*/,
    -9,-1, -7,-6/*mean (0.200967), correlation (0.33312)*/,
    -2,-5, 0,12/*mean (0.201518), correlation (0.340635)*/,
    -12,5, -7,5/*mean (0.207805), correlation (0.335631)*/,
    3,-10, 8,-13/*mean (0.224438), correlation (0.34504)*/,
    -7,-7, -4,5/*mean (0.239361), correlation (0.338053)*/,
    -3,-2, -1,-7/*mean (0.240744), correlation (0.344322)*/,
    2,9, 5,-11/*mean (0.242949), correlation (0.34145)*/,
    -11,-13, -5,-13/*mean (0.244028), correlation (0.336861)*/,
    -1,6, 0,-1/*mean (0.247571), correlation (0.343684)*/,
    5,-3, 5,2/*mean (0.000697256), correlation (0.357265)*/,
    -4,-13, -4,12/*mean (0.00213675), correlation (0.373827)*/,
    -9,-6, -9,6/*mean (0.0126856), correlation (0.373938)*/,
    -12,-10, -8,-4/*mean (0.0152497), correlation (0.364237)*/,
    10,2, 12,-3/*mean (0.0299933), correlation (0.345292)*/,
    7,12, 12,12/*mean (0.0307242), correlation (0.366299)*/,
    -7,-13, -6,5/*mean (0.0534975), correlation (0.368357)*/,
    -4,9, -3,4/*mean (0.099865), correlation (0.372276)*/,
    7,-1, 12,2/*mean (0.117083), correlation (0.364529)*/,
    -7,6, -5,1/*mean (0.126125), correlation (0.369606)*/,
    -13,11, -12,5/*mean (0.130364), correlation (0.358502)*/,
    -3,7, -2,-6/*mean (0.131691), correlation (0.375531)*/,
    7,-8, 12,-7/*mean (0.160166), correlation (0.379508)*/,
    -13,-7, -11,-12/*mean (0.167848), correlation (0.353343)*/,
    1,-3, 12,12/*mean (0.183378), correlation (0.371916)*/,
    2,-6, 3,0/*mean (0.228711), correlation (0.371761)*/,
    -4,3, -2,-13/*mean (0.247211), correlation (0.364063)*/,
    -1,-13, 1,9/*mean (0.249325), correlation (0.378139)*/,
    7,1, 8,-6/*mean (0.000652272), correlation (0.411682)*/,
    1,-1, 3,12/*mean (0.00248538), correlation (0.392988)*/,
    9,1, 12,6/*mean (0.0206815), correlation (0.386106)*/,
    -1,-9, -1,3/*mean (0.0364485), correlation (0.410752)*/,
    -13,-13, -10,5/*mean (0.0376068), correlation (0.398374)*/,
    7,7, 10,12/*mean (0.0424202), correlation (0.405663)*/,
    12,-5, 12,9/*mean (0.0942645), correlation (0.410422)*/,
    6,3, 7,11/*mean (0.1074), correlation (0.413224)*/,
    5,-13, 6,10/*mean (0.109256), correlation (0.408646)*/,
    2,-12, 2,3/*mean (0.131691), correlation (0.416076)*/,
    3,8, 4,-6/*mean (0.165081), correlation (0.417569)*/,
    2,6, 12,-13/*mean (0.171874), correlation (0.408471)*/,
    9,-12, 10,3/*mean (0.175146), correlation (0.41296)*/,
    -8,4, -7,9/*mean (0.183682), correlation (0.402956)*/,
    -11,12, -4,-6/*mean (0.184672), correlation (0.416125)*/,
    1,12, 2,-8/*mean (0.191487), correlation (0.386696)*/,
    6,-9, 7,-4/*mean (0.192668), correlation (0.394771)*/,
    2,3, 3,-2/*mean (0.200157), correlation (0.408303)*/,
    6,3, 11,0/*mean (0.204588), correlation (0.411762)*/,
    3,-3, 8,-8/*mean (0.205904), correlation (0.416294)*/,
    7,8, 9,3/*mean (0.213237), correlation (0.409306)*/,
    -11,-5, -6,-4/*mean (0.243444), correlation (0.395069)*/,
    -10,11, -5,10/*mean (0.247672), correlation (0.413392)*/,
    -5,-8, -3,12/*mean (0.24774), correlation (0.411416)*/,
    -10,5, -9,0/*mean (0.00213675), correlation (0.454003)*/,
    8,-1, 12,-6/*mean (0.0293635), correlation (0.455368)*/,
    4,-6, 6,-11/*mean (0.0404971), correlation (0.457393)*/,
    -10,12, -8,7/*mean (0.0481107), correlation (0.448364)*/,
    4,-2, 6,7/*mean (0.050641), correlation (0.455019)*/,
    -2,0, -2,12/*mean (0.0525978), correlation (0.44338)*/,
    -5,-8, -5,2/*mean (0.0629667), correlation (0.457096)*/,
    7,-6, 10,12/*mean (0.0653846), correlation (0.445623)*/,
    -9,-13, -8,-8/*mean (0.0858749), correlation (0.449789)*/,
    -5,-13, -5,-2/*mean (0.122402), correlation (0.450201)*/,
    8,-8, 9,-13/*mean (0.125416), correlation (0.453224)*/,
    -9,-11, -9,0/*mean (0.130128), correlation (0.458724)*/,
    1,-8, 1,-2/*mean (0.132467), correlation (0.440133)*/,
    7,-4, 9,1/*mean (0.132692), correlation (0.454)*/,
    -2,1, -1,-4/*mean (0.135695), correlation (0.455739)*/,
    11,-6, 12,-11/*mean (0.142904), correlation (0.446114)*/,
    -12,-9, -6,4/*mean (0.146165), correlation (0.451473)*/,
    3,7, 7,12/*mean (0.147627), correlation (0.456643)*/,
    5,5, 10,8/*mean (0.152901), correlation (0.455036)*/,
    0,-4, 2,8/*mean (0.167083), correlation (0.459315)*/,
    -9,12, -5,-13/*mean (0.173234), correlation (0.454706)*/,
    0,7, 2,12/*mean (0.18312), correlation (0.433855)*/,
    -1,2, 1,7/*mean (0.185504), correlation (0.443838)*/,
    5,11, 7,-9/*mean (0.185706), correlation (0.451123)*/,
    3,5, 6,-8/*mean (0.188968), correlation (0.455808)*/,
    -13,-4, -8,9/*mean (0.191667), correlation (0.459128)*/,
    -5,9, -3,-3/*mean (0.193196), correlation (0.458364)*/,
    -4,-7, -3,-12/*mean (0.196536), correlation (0.455782)*/,
    6,5, 8,0/*mean (0.1972), correlation (0.450481)*/,
    -7,6, -6,12/*mean (0.199438), correlation (0.458156)*/,
    -13,6, -5,-2/*mean (0.211224), correlation (0.449548)*/,
    1,-10, 3,10/*mean (0.211718), correlation (0.440606)*/,
    4,1, 8,-4/*mean (0.213034), correlation (0.443177)*/,
    -2,-2, 2,-13/*mean (0.234334), correlation (0.455304)*/,
    2,-12, 12,12/*mean (0.235684), correlation (0.443436)*/,
    -2,-13, 0,-6/*mean (0.237674), correlation (0.452525)*/,
    4,1, 9,3/*mean (0.23962), correlation (0.444824)*/,
    -6,-10, -3,-5/*mean (0.248459), correlation (0.439621)*/,
    -3,-13, -1,1/*mean (0.249505), correlation (0.456666)*/,
    7,5, 12,-11/*mean (0.00119208), correlation (0.495466)*/,
    4,-2, 5,-7/*mean (0.00372245), correlation (0.484214)*/,
    -13,9, -9,-5/*mean (0.00741116), correlation (0.499854)*/,
    7,1, 8,6/*mean (0.0208952), correlation (0.499773)*/,
    7,-8, 7,6/*mean (0.0220085), correlation (0.501609)*/,
    -7,-4, -7,1/*mean (0.0233806), correlation (0.496568)*/,
    -8,11, -7,-8/*mean (0.0236505), correlation (0.489719)*/,
    -13,6, -12,-8/*mean (0.0268781), correlation (0.503487)*/,
    2,4, 3,9/*mean (0.0323324), correlation (0.501938)*/,
    10,-5, 12,3/*mean (0.0399235), correlation (0.494029)*/,
    -6,-5, -6,7/*mean (0.0420153), correlation (0.486579)*/,
    8,-3, 9,-8/*mean (0.0548021), correlation (0.484237)*/,
    2,-12, 2,8/*mean (0.0616622), correlation (0.496642)*/,
    -11,-2, -10,3/*mean (0.0627755), correlation (0.498563)*/,
    -12,-13, -7,-9/*mean (0.0829622), correlation (0.495491)*/,
    -11,0, -10,-5/*mean (0.0843342), correlation (0.487146)*/,
    5,-3, 11,8/*mean (0.0929937), correlation (0.502315)*/,
    -2,-13, -1,12/*mean (0.113327), correlation (0.48941)*/,
    -1,-8, 0,9/*mean (0.132119), correlation (0.467268)*/,
    -13,-11, -12,-5/*mean (0.136269), correlation (0.498771)*/,
    -10,-2, -10,11/*mean (0.142173), correlation (0.498714)*/,
    -3,9, -2,-13/*mean (0.144141), correlation (0.491973)*/,
    2,-3, 3,2/*mean (0.14892), correlation (0.500782)*/,
    -9,-13, -4,0/*mean (0.150371), correlation (0.498211)*/,
    -4,6, -3,-10/*mean (0.152159), correlation (0.495547)*/,
    -4,12, -2,-7/*mean (0.156152), correlation (0.496925)*/,
    -6,-11, -4,9/*mean (0.15749), correlation (0.499222)*/,
    6,-3, 6,11/*mean (0.159211), correlation (0.503821)*/,
    -13,11, -5,5/*mean (0.162427), correlation (0.501907)*/,
    11,11, 12,6/*mean (0.16652), correlation (0.497632)*/,
    7,-5, 12,-2/*mean (0.169141), correlation (0.484474)*/,
    -1,12, 0,7/*mean (0.169456), correlation (0.495339)*/,
    -4,-8, -3,-2/*mean (0.171457), correlation (0.487251)*/,
    -7,1, -6,7/*mean (0.175), correlation (0.500024)*/,
    -13,-12, -8,-13/*mean (0.175866), correlation (0.497523)*/,
    -7,-2, -6,-8/*mean (0.178273), correlation (0.501854)*/,
    -8,5, -6,-9/*mean (0.181107), correlation (0.494888)*/,
    -5,-1, -4,5/*mean (0.190227), correlation (0.482557)*/,
    -13,7, -8,10/*mean (0.196739), correlation (0.496503)*/,
    1,5, 5,-13/*mean (0.19973), correlation (0.499759)*/,
    1,0, 10,-13/*mean (0.204465), correlation (0.49873)*/,
    9,12, 10,-1/*mean (0.209334), correlation (0.49063)*/,
    5,-8, 10,-9/*mean (0.211134), correlation (0.503011)*/,
    -1,11, 1,-13/*mean (0.212), correlation (0.499414)*/,
    -9,-3, -6,2/*mean (0.212168), correlation (0.480739)*/,
    -1,-10, 1,12/*mean (0.212731), correlation (0.502523)*/,
    -13,1, -8,-10/*mean (0.21327), correlation (0.489786)*/,
    8,-11, 10,-6/*mean (0.214159), correlation (0.488246)*/,
    2,-13, 3,-6/*mean (0.216993), correlation (0.50287)*/,
    7,-13, 12,-9/*mean (0.223639), correlation (0.470502)*/,
    -10,-10, -5,-7/*mean (0.224089), correlation (0.500852)*/,
    -10,-8, -8,-13/*mean (0.228666), correlation (0.502629)*/,
    4,-6, 8,5/*mean (0.22906), correlation (0.498305)*/,
    3,12, 8,-13/*mean (0.233378), correlation (0.503825)*/,
    -4,2, -3,-3/*mean (0.234323), correlation (0.476692)*/,
    5,-13, 10,-12/*mean (0.236392), correlation (0.475462)*/,
    4,-13, 5,-1/*mean (0.236842), correlation (0.504132)*/,
    -9,9, -4,3/*mean (0.236977), correlation (0.497739)*/,
    0,3, 3,-9/*mean (0.24314), correlation (0.499398)*/,
    -12,1, -6,1/*mean (0.243297), correlation (0.489447)*/,
    3,2, 4,-8/*mean (0.00155196), correlation (0.553496)*/,
    -10,-10, -10,9/*mean (0.00239541), correlation (0.54297)*/,
    8,-13, 12,12/*mean (0.0034413), correlation (0.544361)*/,
    -8,-12, -6,-5/*mean (0.003565), correlation (0.551225)*/,
    2,2, 3,7/*mean (0.00835583), correlation (0.55285)*/,
    10,6, 11,-8/*mean (0.00885065), correlation (0.540913)*/,
    6,8, 8,-12/*mean (0.0101552), correlation (0.551085)*/,
    -7,10, -6,5/*mean (0.0102227), correlation (0.533635)*/,
    -3,-9, -3,9/*mean (0.0110211), correlation (0.543121)*/,
    -1,-13, -1,5/*mean (0.0113473), correlation (0.550173)*/,
    -3,-7, -3,4/*mean (0.0140913), correlation (0.554774)*/,
    -8,-2, -8,3/*mean (0.017049), correlation (0.55461)*/,
    4,2, 12,12/*mean (0.01778), correlation (0.546921)*/,
    2,-5, 3,11/*mean (0.0224022), correlation (0.549667)*/,
    6,-9, 11,-13/*mean (0.029161), correlation (0.546295)*/,
    3,-1, 7,12/*mean (0.0303081), correlation (0.548599)*/,
    11,-1, 12,4/*mean (0.0355151), correlation (0.523943)*/,
    -3,0, -3,6/*mean (0.0417904), correlation (0.543395)*/,
    4,-11, 4,12/*mean (0.0487292), correlation (0.542818)*/,
    2,-4, 2,1/*mean (0.0575124), correlation (0.554888)*/,
    -10,-6, -8,1/*mean (0.0594242), correlation (0.544026)*/,
    -13,7, -11,1/*mean (0.0597391), correlation (0.550524)*/,
    -13,12, -11,-13/*mean (0.0608974), correlation (0.55383)*/,
    6,0, 11,-13/*mean (0.065126), correlation (0.552006)*/,
    0,-1, 1,4/*mean (0.074224), correlation (0.546372)*/,
    -13,3, -9,-2/*mean (0.0808592), correlation (0.554875)*/,
    -9,8, -6,-3/*mean (0.0883378), correlation (0.551178)*/,
    -13,-6, -8,-2/*mean (0.0901035), correlation (0.548446)*/,
    5,-9, 8,10/*mean (0.0949843), correlation (0.554694)*/,
    2,7, 3,-9/*mean (0.0994152), correlation (0.550979)*/,
    -1,-6, -1,-1/*mean (0.10045), correlation (0.552714)*/,
    9,5, 11,-2/*mean (0.100686), correlation (0.552594)*/,
    11,-3, 12,-8/*mean (0.101091), correlation (0.532394)*/,
    3,0, 3,5/*mean (0.101147), correlation (0.525576)*/,
    -1,4, 0,10/*mean (0.105263), correlation (0.531498)*/,
    3,-6, 4,5/*mean (0.110785), correlation (0.540491)*/,
    -13,0, -10,5/*mean (0.112798), correlation (0.536582)*/,
    5,8, 12,11/*mean (0.114181), correlation (0.555793)*/,
    8,9, 9,-6/*mean (0.117431), correlation (0.553763)*/,
    7,-4, 8,-12/*mean (0.118522), correlation (0.553452)*/,
    -10,4, -10,9/*mean (0.12094), correlation (0.554785)*/,
    7,3, 12,4/*mean (0.122582), correlation (0.555825)*/,
    9,-7, 10,-2/*mean (0.124978), correlation (0.549846)*/,
    7,0, 12,-2/*mean (0.127002), correlation (0.537452)*/,
    -1,-6, 0,-11/*mean (0.127148), correlation (0.547401)*/
};

/**
 * !特征提取器的构造函数
 * todo step1: 计算图像金字塔每一层的缩放因子
 * todo step2: 计算图像金字塔每一层要提取的特征点数目
 * todo step3: 计算灰度心圆中，每一行像素的umax值
 */ 
ORBextractor::ORBextractor(
    int _nfeatures,         //指定要提取的特征点的数目
    float _scaleFactor,     //指定图像金字塔的缩放系数
    int _nlevels,           //指定图像金字塔的层数
    int _iniThFAST,         //指定提取ORB特帧中的fast关键点的默认阈值
    int _minThFAST          /*如果默认阈值提取不出足够的特征点(图像文理不丰富)，为了达到想要的特征点数目，
                            就使用这个参数提取不是那么明显的角点*/
    ):
        nfeatures(_nfeatures),      //初始化这些参数
        scaleFactor(_scaleFactor), 
        nlevels(_nlevels),
        iniThFAST(_iniThFAST), 
        minThFAST(_minThFAST)
{   
    /**
     * ?step1: 计算图像金字塔每一层的缩放因子
    */
    //存储每层图像缩放因子的vector，将其大小调整与图层的数目相同
    mvScaleFactor.resize(nlevels);
    //存储这个sigma^2的vector，其实就是每层图像相对于初始图像缩放因子的平方，将其大小调整与图层的数目相同
    mvLevelSigma2.resize(nlevels);
    //对于初始图像，即第一层图像，这两个参数(缩放因子，sigma)的值都为1
    mvScaleFactor[0]=1.0f;
    mvLevelSigma2[0]=1.0f;

    //然后逐层计算图像金字塔中图像相当于初始图像的缩放系数
    for(int i=1; i<nlevels; i++)
    {   
        //scaleFactor配置文件中设置为 1.2 
        //类乘 这一层的缩放因子 = 上一层的缩放因子 * 1.2
        mvScaleFactor[i]=mvScaleFactor[i-1]*scaleFactor;
        //当前层的sigma=当前层的缩放因子的平方
        mvLevelSigma2[i]=mvScaleFactor[i]*mvScaleFactor[i];
    }
    //这两个向量(vector)保留上面的参数的到数
    mvInvScaleFactor.resize(nlevels);
    mvInvLevelSigma2.resize(nlevels);
    for(int i=0; i<nlevels; i++)
    {
        mvInvScaleFactor[i]=1.0f/mvScaleFactor[i]; //取倒数
        mvInvLevelSigma2[i]=1.0f/mvLevelSigma2[i];
    }

    mvImagePyramid.resize(nlevels);



    /**
     * ?step2: 计算图像金字塔每一层要提取的特征点数目
     * ?计算原理在课件2.5中P18
    */
    //mnFeaturesPerLevel 存放每一层要提取的特征点数目
    mnFeaturesPerLevel.resize(nlevels);
    float factor = 1.0f / scaleFactor;
    float nDesiredFeaturesPerScale = nfeatures*(1 - factor)/(1 - (float)pow((double)factor, (double)nlevels));

    int sumFeatures = 0;    //统计第0层到到数第二层的分配的特征点数目
    for( int level = 0; level < nlevels-1; level++ )
    {
        mnFeaturesPerLevel[level] = cvRound(nDesiredFeaturesPerScale);
        sumFeatures += mnFeaturesPerLevel[level];
        nDesiredFeaturesPerScale *= factor;
    }
    //分配给最后一层的特征点数目 = 总提取数 - 已分配给第0层到倒数第二层的特征点数目
    mnFeaturesPerLevel[nlevels-1] = std::max(nfeatures - sumFeatures, 0);   

    //计算描述子需要的点的个数。上面数组中(256*2*2)存储的是坐标,256bit的描述子
    const int npoints = 512;
    /**
     * !获取计算BRIEF描述子的随即采样点点集的指针
     * pattern0是Point*，bit_pattern_31_是int类型，所以需要强制转换
     */
    const Point* pattern0 = (const Point*)bit_pattern_31_;
    /**
     * 上面一句代码和这一句代码的操作：将全局变量区域中int格式的随即采样点集以cv::point格式复制到当前类对象的成员变量中
    */
    std::copy(pattern0, pattern0 + npoints, std::back_inserter(pattern));

    


    
    //This is for orientation
    // pre-compute the end of a row in a circular patch
    /**
     * ?step3: 计算灰度心圆中，每一行像素的umax值，计算特征点方向信息时用到
     * 预先计算圆形patch中行的结束位置
     * umax：vector类型 HALF_PATCH_SIZE的值为15，即半径为15，所以umax的长度为16
     * umax：1/4圆的每一行的u轴坐标边界
     * HALF_PATCH_SIZE=15
    */
    umax.resize(HALF_PATCH_SIZE + 1);

    /**
     * cvFloor函数：返回不大于参数的最大整数值，即向下取正
     * cvCeil函数：返回不小于参数的最小整数值，即向上取正
     * cvRound函数：四舍五入
     * ?2.3 原理图
    */
    int v, 
        v0, 
        vmax = cvFloor(HALF_PATCH_SIZE * sqrt(2.f) / 2 + 1);
    //vmax和vmin相差1
    int vmin = cvCeil(HALF_PATCH_SIZE * sqrt(2.f) / 2);
    //hp2=半径的平方
    const double hp2 = HALF_PATCH_SIZE*HALF_PATCH_SIZE;

    //利用圆的方程(或者是勾股定理)，计算每一行像素u坐标的边界
    for (v = 0; v <= vmax; ++v)
        umax[v] = cvRound(sqrt(hp2 - v * v));

    // Make sure we are symmetric
    //这里使用对称的方式计算上八分之一圆周上的umax，目的也是为了保持严格的对陈
    //如果按照常规的想法做，由于cvRound就会很容易出现不对称的情况，同时这些随即采样的特征点集也不能够满足旋转不变性了
    for (v = HALF_PATCH_SIZE, v0 = 0; v >= vmin; --v)
    {
        while (umax[v0] == umax[v0 + 1])
            ++v0;
        umax[v] = v0;
        ++v0;
    }
}



/**
 * 计算特征点的方向,实现特征不变性
 * @param1  对应的图层的图像
 * @param2  这个图层中剔除后的特征点的容器
 * @param3  横坐标边界
*/
static void computeOrientation(const Mat& image, vector<KeyPoint>& keypoints, const vector<int>& umax)
{
    //在该层图像循环遍历每一个角点 计算方向信息
    for (vector<KeyPoint>::iterator keypoint = keypoints.begin(),
         keypointEnd = keypoints.end(); keypoint != keypointEnd; ++keypoint)
    {
        keypoint->angle = IC_Angle(
            image,          //特征点所在的图像
            keypoint->pt,   //特征点的坐标
            umax            //每个特征点所在的图像区块的每行的边界 vector容器
        );
    }
}



void ExtractorNode::DivideNode(ExtractorNode &n1, ExtractorNode &n2, ExtractorNode &n3, ExtractorNode &n4)
{
    const int halfX = ceil(static_cast<float>(UR.x-UL.x)/2);
    const int halfY = ceil(static_cast<float>(BR.y-UL.y)/2);

    /**
     * 分别计算n1 n2 n3 n4四个特征提取器节点的边界坐标
     * ?原理图P20
    */
    //Define boundaries of childs
    n1.UL = UL;
    n1.UR = cv::Point2i(UL.x+halfX,UL.y);
    n1.BL = cv::Point2i(UL.x,UL.y+halfY);
    n1.BR = cv::Point2i(UL.x+halfX,UL.y+halfY);
    n1.vKeys.reserve(vKeys.size());

    n2.UL = n1.UR;
    n2.UR = UR;
    n2.BL = n1.BR;
    n2.BR = cv::Point2i(UR.x,UL.y+halfY);
    n2.vKeys.reserve(vKeys.size());

    n3.UL = n1.BL;
    n3.UR = n1.BR;
    n3.BL = BL;
    n3.BR = cv::Point2i(n1.BR.x,BL.y);
    n3.vKeys.reserve(vKeys.size());

    n4.UL = n3.UR;
    n4.UR = n2.BR;
    n4.BL = n3.BR;
    n4.BR = BR;
    n4.vKeys.reserve(vKeys.size());

    //循环将父特征提取器节点中的特征点 分配给四个子特征提取器节点中
    for(size_t i=0;i<vKeys.size();i++)
    {
        const cv::KeyPoint &kp = vKeys[i];

        //判断当前遍历到的特征点是属于哪一个特征提取器节点的区域中，属于哪一个子块
        if(kp.pt.x<n1.UR.x)
        {
            if(kp.pt.y<n1.BR.y)
                n1.vKeys.push_back(kp);
            else
                n3.vKeys.push_back(kp);
        }
        else if(kp.pt.y<n1.BR.y)
            n2.vKeys.push_back(kp);
        else
            n4.vKeys.push_back(kp);
    }

    //若这四个区域中只有一个特征点，则作号标记
    if(n1.vKeys.size()==1)
        n1.bNoMore = true;
    if(n2.vKeys.size()==1)
        n2.bNoMore = true;
    if(n3.vKeys.size()==1)
        n3.bNoMore = true;
    if(n4.vKeys.size()==1)
        n4.bNoMore = true;

}


/**
 * 使用4叉树法对一个图层中的特征点进行平均和划分
 * ?原理 P18～19
*/
vector<cv::KeyPoint> ORBextractor::DistributeOctTree( //返回值是一个保存剔除后特征点的容器
    const vector<cv::KeyPoint>& vToDistributeKeys,  //等待进行分配到4叉树的特征点
    const int &minX,                                //注意：这里的坐标都是在"半径扩充图像"下的坐标
    const int &maxX,        
    const int &minY, 
    const int &maxY, 
    const int &N,                                   //希望取出的特征点数目
    const int &level)                               //指定的图层，但是在本函数中，并没有使用
{//这个函数直接使用类自己的成员变量，图像金字塔中的图像

    
    //计算应该生成的初始节点个数。根据边界的高宽比值确定初始节点个数。
    const int nIni = round(static_cast<float>(maxX-minX)/(maxY-minY));

    //初始节点x方向有多少个像素（有多宽）
    const float hX = static_cast<float>(maxX-minX)/nIni;

    //存储 提取器节点 的列表
    list<ExtractorNode> lNodes;

    //存储 初始提取器节点指针 的列表
    vector<ExtractorNode*> vpIniNodes;
    vpIniNodes.resize(nIni);

    //生成 nIni 个初始提取器节点
    for(int i=0; i<nIni; i++)
    {   
        //声明一个提取器节点
        ExtractorNode ni;

        //设置提取器节点的图像边界。节点的边界先按照“边缘扩充图像“下的坐标来理解
        ni.UL = cv::Point2i(hX*static_cast<float>(i),0);    //UpLeft
        ni.UR = cv::Point2i(hX*static_cast<float>(i+1),0);  //Upight
        ni.BL = cv::Point2i(ni.UL.x,maxY-minY);             //BottomLeft
        ni.BR = cv::Point2i(ni.UR.x,maxY-minY);             //BottomRight
        ni.vKeys.reserve(vToDistributeKeys.size());

        //将初始提取器节点添加到lNodes列表中
        lNodes.push_back(ni);
        //将初始的提取器节点句柄添加到vpIniNodes列表中
        vpIniNodes[i] = &lNodes.back();
    }

 
    //将特征点分配到初始提取器节点中
    for(size_t i=0;i<vToDistributeKeys.size();i++)//开始遍历等待分配的提取器节点
    {
        //获取特征点对象
        const cv::KeyPoint &kp = vToDistributeKeys[i];
        //按特征点的横轴位置，分配给所属的那个图像区域的初始提取器节点中
        vpIniNodes[kp.pt.x/hX]->vKeys.push_back(kp);
    }

    //遍历lNodes列表(存储提取器节点列表)，标记那些不可再分裂的节点，删除那些没有分配到特征点的节点
    list<ExtractorNode>::iterator lit = lNodes.begin();
    while(lit!=lNodes.end())
    {
        //如果初始提取器节点所分配的特征点个数为1
        if(lit->vKeys.size()==1)
        {
            //作标记，表示不可再分
            lit->bNoMore=true;
            lit++;
        }
        //如果一个提取器节点没有被分配到特征点，那么从列表中直接剔除它
        else if(lit->vKeys.empty())
            lit = lNodes.erase(lit);
        //不是以上两种情况则遍历下一个
        else
            lit++;
    }//这一层循环遍历的是lNodes列表中所有的初始提取器节点

    //结束标志为false
    bool bFinish = false;
    //迭代次数
    int iteration = 0;

    //用于存储vsize和句柄对的列表
    //这个变量记录了在一次分裂循环中，那些可以再继续进行分裂的节点中包含的特征点数和其句柄
    vector<pair<int,ExtractorNode*> > vSizeAndPointerToNode;
    //调整其大小，将每一个初始化节点分裂成4个，
    vSizeAndPointerToNode.reserve(lNodes.size()*4);

    //根据特征点分布，利用N叉树对图像进行划分区域
    while(!bFinish)
    {   
        //迭代次数加+1，本函数没有用到该局部变量
        iteration++;

        //保留当前节点数
        int prevSize = lNodes.size();

        //重新定位迭代器指向列表头部
        lit = lNodes.begin();

        //需要展开的节点计数
        int nToExpand = 0;

        //因为在循环体中，前一轮的循环可能污染了该变量，所以清空
        vSizeAndPointerToNode.clear();

        //将目前的子区域进行划分
        //开始遍历列表中，所有提取器节点，并进行分解和保留
        while(lit!=lNodes.end())
        {
            //如果提取器节点只有一个特征点
            if(lit->bNoMore)
            {
                //没有必要在进行细分，跳过该节点
                lit++;
                continue;
            }
            else
            {
                //如果提取器节点超过一个特征点，则继续细分
                ExtractorNode n1,n2,n3,n4;
                lit->DivideNode(n1,n2,n3,n4);   //细分为四个子区域

                //该区域有特征点，则添加到提取器节点列表中
                if(n1.vKeys.size()>0)
                {
                    lNodes.push_front(n1);
                    //如果超过一个特征点                    
                    if(n1.vKeys.size()>1)
                    {   
                        //待展开节点计数+1
                        nToExpand++;
                        //保存这个区域特征点数目和节点指针信息
                        vSizeAndPointerToNode.push_back(make_pair(n1.vKeys.size(),&lNodes.front()));
                        //迭代
                        lNodes.front().lit = lNodes.begin();
                    }
                }

                //操作与n1相同
                if(n2.vKeys.size()>0)
                {
                    lNodes.push_front(n2);
                    if(n2.vKeys.size()>1)
                    {
                        nToExpand++;
                        vSizeAndPointerToNode.push_back(make_pair(n2.vKeys.size(),&lNodes.front()));
                        lNodes.front().lit = lNodes.begin();
                    }
                }
                //操作与n1相同
                if(n3.vKeys.size()>0)
                {
                    lNodes.push_front(n3);
                    if(n3.vKeys.size()>1)
                    {
                        nToExpand++;
                        vSizeAndPointerToNode.push_back(make_pair(n3.vKeys.size(),&lNodes.front()));
                        lNodes.front().lit = lNodes.begin();
                    }
                }
                //操作与n1相同
                if(n4.vKeys.size()>0)
                {
                    lNodes.push_front(n4);
                    if(n4.vKeys.size()>1)
                    {
                        nToExpand++;
                        vSizeAndPointerToNode.push_back(make_pair(n4.vKeys.size(),&lNodes.front()));
                        lNodes.front().lit = lNodes.begin();
                    }
                }

                //删除父节点
                lit=lNodes.erase(lit);
                continue;
            }
        }//这一层while循环对每一个初始特征节点提取器进行分裂       
        
       
        //如果分裂的区域大于我们希望取出的特征点数目,则没有必要再分了
        if((int)lNodes.size()>=N || (int)lNodes.size()==prevSize)
        {
            bFinish = true;
        }

        //当再次划分，所有的提取器节点数大于我们希望取出的特征点数目N时，就逐个慢慢划分，直到刚好后者刚超过
        else if(((int)lNodes.size()+nToExpand*3)>N)
        {

            while(!bFinish)
            {
                //保留当前节点数
                prevSize = lNodes.size();
                //用于存储vsize和句柄对的列表
                //这个变量记录了在一次分裂循环中，那些可以再继续进行分裂的节点中包含的特征点数和其句柄
                vector<pair<int,ExtractorNode*> > vPrevSizeAndPointerToNode = vSizeAndPointerToNode;
                //因为在循环体中，前一轮的循环可能污染了该变量，所以清空
                vSizeAndPointerToNode.clear();

                //对需要划分的部分进行排序，对pair对的第一个元素进行排序，特征点越多的部分排越后
                //而for循环是从后往前遍历的，所以特征点稀疏的区域被分裂的几率更小，而密集区被分裂的几率更大
                sort(vPrevSizeAndPointerToNode.begin(),vPrevSizeAndPointerToNode.end());
                for(int j=vPrevSizeAndPointerToNode.size()-1;j>=0;j--)
                {
                    ExtractorNode n1,n2,n3,n4;
                    //对每个需要分裂的节点进行分裂
                    vPrevSizeAndPointerToNode[j].second->DivideNode(n1,n2,n3,n4);

                    // 接下来的操作逻辑与前面的一样
                    if(n1.vKeys.size()>0)
                    {
                        lNodes.push_front(n1);
                        if(n1.vKeys.size()>1)
                        {
                            vSizeAndPointerToNode.push_back(make_pair(n1.vKeys.size(),&lNodes.front()));
                            lNodes.front().lit = lNodes.begin();
                        }
                    }
                    if(n2.vKeys.size()>0)
                    {
                        lNodes.push_front(n2);
                        if(n2.vKeys.size()>1)
                        {
                            vSizeAndPointerToNode.push_back(make_pair(n2.vKeys.size(),&lNodes.front()));
                            lNodes.front().lit = lNodes.begin();
                        }
                    }
                    if(n3.vKeys.size()>0)
                    {
                        lNodes.push_front(n3);
                        if(n3.vKeys.size()>1)
                        {
                            vSizeAndPointerToNode.push_back(make_pair(n3.vKeys.size(),&lNodes.front()));
                            lNodes.front().lit = lNodes.begin();
                        }
                    }
                    if(n4.vKeys.size()>0)
                    {
                        lNodes.push_front(n4);
                        if(n4.vKeys.size()>1)
                        {
                            vSizeAndPointerToNode.push_back(make_pair(n4.vKeys.size(),&lNodes.front()));
                            lNodes.front().lit = lNodes.begin();
                        }
                    }

                    lNodes.erase(vPrevSizeAndPointerToNode[j].second->lit);
                     //如果分裂的区域大于我们希望取出的特征点数目,则没有必要再分了
                    if((int)lNodes.size()>=N)
                        break;
                }
                
                if((int)lNodes.size()>=N || (int)lNodes.size()==prevSize)
                    bFinish = true;

            }
        }
    }

    //分列完之后，每一个区域要保留一个区域响应值最大的关键点
    //存储保留结果
    vector<cv::KeyPoint> vResultKeys;
    vResultKeys.reserve(nfeatures);

    //对提取器节点区域进行遍历
    for(list<ExtractorNode>::iterator lit=lNodes.begin(); lit!=lNodes.end(); lit++)
    {
        //得到这个节点区域中特征点的句柄
        vector<cv::KeyPoint> &vNodeKeys = lit->vKeys;
        //得到指向第一个特征点的指针
        cv::KeyPoint* pKP = &vNodeKeys[0];
        //初始化最大响应值
        float maxResponse = pKP->response;

        //对该节点区域的特征点进行遍历，找出响应值最大的关键点
        for(size_t k=1;k<vNodeKeys.size();k++)
        {
            if(vNodeKeys[k].response>maxResponse)
            {
                pKP = &vNodeKeys[k];
                maxResponse = vNodeKeys[k].response;
            }
        }

        //将遍历到的每一个节点区域中的响应值最大的关键点添加进结果
        vResultKeys.push_back(*pKP);
    }
    //返回结果
    return vResultKeys;
}

/**
 * 特征点的提取和分配
*/
void ORBextractor::ComputeKeyPointsOctTree(
    vector<vector<KeyPoint> >& allKeypoints //传出参数
    )                                       
                                           
{
    allKeypoints.resize(nlevels);

    //*step1：将图像分成若干个 30x30 的网格
    const float W = 30;

    //遍历每一层图像
    for (int level = 0; level < nlevels; ++level)
    {
        //计算这层图像的像素坐标边界 //?看原理图P17
        const int minBorderX = EDGE_THRESHOLD-3;    //减3，是因为需要建立一个半径为3的圆
        const int minBorderY = minBorderX;
        const int maxBorderX = mvImagePyramid[level].cols-EDGE_THRESHOLD+3;
        const int maxBorderY = mvImagePyramid[level].rows-EDGE_THRESHOLD+3;

        //存储需要进行平均分配的特征点
        vector<cv::KeyPoint> vToDistributeKeys;
        //先过量采集 需要nfeatures  采集nfeatures*10
        vToDistributeKeys.reserve(nfeatures*10);

        //计算进行特帧提取的图像区域
        const float width = (maxBorderX-minBorderX);
        const float height = (maxBorderY-minBorderY);

        //计算图像每一行(列)有多少个网格
        const int nCols = width/W;
        const int nRows = height/W;
        //计算每个图像网格所占的像素行数和列数
        const int wCell = ceil(width/nCols);
        const int hCell = ceil(height/nRows);

        //按行开始遍历网格
        for(int i=0; i<nRows; i++)
        {   
            //计算当前网格的初始行坐标
            const float iniY =minBorderY+i*hCell;
            //计算当前网格的最大行坐标。(6=3+3 包括多出来的，方便计算FAST特征点的3像素)
            float maxY = iniY+hCell+6;

            //如果初始的行坐标大于图像边界，则跳过此次循环
            if(iniY>=maxBorderY-3)
                continue;
            //如果图像的大小不能恰好划分出整齐的网格，则牺牲本行最后一个网格
            if(maxY>maxBorderY)
                maxY = maxBorderY;

            //开始列遍历
            for(int j=0; j<nCols; j++)
            {
                //计算当前网格的初始列坐标
                const float iniX =minBorderX+j*wCell;
                //计算当前网格的最大列坐标。(6=3+3 包括多出来的，方便计算FAST特征点的3像素)
                float maxX = iniX+wCell+6;
                //如果初始的列坐标大于图像边界，则跳过此次循环
                if(iniX>=maxBorderX-6)//!疑似bug
                    continue;
                //如果图像的大小不能恰好划分出整齐的网格，则牺牲本列最后一个网格
                if(maxX>maxBorderX)
                    maxX = maxBorderX;

                //这个向量存储这个网格的关键点
                vector<cv::KeyPoint> vKeysCell;
                //!调用OpenCV的库函数
                FAST(mvImagePyramid[level].rowRange(iniY,maxY).colRange(iniX,maxX), //代提取的网格
                     vKeysCell,     //存储关键点的容器 
                     iniThFAST,     //检测的阈值
                     true           //使能非极大值抑制
                );
                    
                //*step2：遍历所有网格，在网格中提取FAST关键点
                //如果使用阈值iniThFAST不能检测出关键点，则使用更低的阈值来重新进行检测
                if(vKeysCell.empty())
                {
                    FAST(mvImagePyramid[level].rowRange(iniY,maxY).colRange(iniX,maxX), //代提取的网格
                         vKeysCell,
                         minThFAST, //更低的阈值
                         true);
                }
                
                //如果提取出了关键点
                if(!vKeysCell.empty())
                {   
                    //遍历所有的FAST角点
                    for(vector<cv::KeyPoint>::iterator vit=vKeysCell.begin(); vit!=vKeysCell.end();vit++)
                    {   
                        //FAST函数提取出来的角点都是基于cell网格的。要将其恢复到在图像中的坐标
                        //*在下面使用4叉树法整理特征点的时候会使用到这个坐标
                        (*vit).pt.x+=j*wCell;
                        (*vit).pt.y+=i*hCell;
                        //然后将其加入到“等待被分配”特征点容器中
                        vToDistributeKeys.push_back(*vit);
                    }
                }

            }
        }//!到这里实现对当前图像的所有网格进行FAST关键点提取

        //声明一个对存储当前图层特征点的容器的引用
        //引用& 所以修改了keypoints 也就修改了allKeypoints[level]
        vector<KeyPoint> & keypoints = allKeypoints[level];
        //调整其大小
        keypoints.reserve(nfeatures);

        //根据mnFeaturesPerLevel(存储每一层图像应提取的特征点数目)，使用4叉树法对多余的特征点进行剔除
        /**
         * 返回值是一个保存特征点的容器，含有经过剔除后保留下来的特征点
         * 得到的特征点的坐标，依然是基于当前图像帧讲的
        */
        //*step3：对每一层的图像剔除特征点，使特征点均匀分布 
        keypoints = DistributeOctTree(
            vToDistributeKeys,          //待剔除的特征点集
            minBorderX, 
            maxBorderX,                 
            minBorderY, 
            maxBorderY,
            mnFeaturesPerLevel[level],  //希望保留下来的当前层图像的特征点数
            level                       //当前层图像所在的层数
        );

        //PATCH_SIZE是基于底层图像来说的，现在要根据当前图层的尺度缩放倍数进行缩放得到缩放后的PATCH大小 和特征点的方向计算相关
        const int scaledPatchSize = PATCH_SIZE*mvScaleFactor[level];

        // Add border to coordinates and scale information
        //剔除后的特征点数目
        const int nkps = keypoints.size();
        //*step4：遍历剔除后的特征，恢复其在当前图层图像坐标系下的坐标
        for(int i=0; i<nkps ; i++)
        {   
            //对于每一个保留下来的特征点，恢复到相对于当前图层“边缘扩充图像下”的坐标系的坐标
            keypoints[i].pt.x+=minBorderX;
            keypoints[i].pt.y+=minBorderY;
            //记录特征点来源的图像的金字塔图层，来自哪一层
            keypoints[i].octave=level;
            //记录计算方向的Patch，又称为特征点半径
            keypoints[i].size = scaledPatchSize;
        }
    }

    //*step5：循环分层计算，计算特征点的方向信息
    for (int level = 0; level < nlevels; ++level)
        computeOrientation(
            mvImagePyramid[level],  //对应的图层的图像
            allKeypoints[level],    //这个图层中剔除后的特征点的容器
            umax                    //横坐标边界
        );
}

void ORBextractor::ComputeKeyPointsOld(std::vector<std::vector<KeyPoint> > &allKeypoints)
{
    allKeypoints.resize(nlevels);

    float imageRatio = (float)mvImagePyramid[0].cols/mvImagePyramid[0].rows;

    for (int level = 0; level < nlevels; ++level)
    {
        const int nDesiredFeatures = mnFeaturesPerLevel[level];

        const int levelCols = sqrt((float)nDesiredFeatures/(5*imageRatio));
        const int levelRows = imageRatio*levelCols;

        const int minBorderX = EDGE_THRESHOLD;
        const int minBorderY = minBorderX;
        const int maxBorderX = mvImagePyramid[level].cols-EDGE_THRESHOLD;
        const int maxBorderY = mvImagePyramid[level].rows-EDGE_THRESHOLD;

        const int W = maxBorderX - minBorderX;
        const int H = maxBorderY - minBorderY;
        const int cellW = ceil((float)W/levelCols);
        const int cellH = ceil((float)H/levelRows);

        const int nCells = levelRows*levelCols;
        const int nfeaturesCell = ceil((float)nDesiredFeatures/nCells);

        vector<vector<vector<KeyPoint> > > cellKeyPoints(levelRows, vector<vector<KeyPoint> >(levelCols));

        vector<vector<int> > nToRetain(levelRows,vector<int>(levelCols,0));
        vector<vector<int> > nTotal(levelRows,vector<int>(levelCols,0));
        vector<vector<bool> > bNoMore(levelRows,vector<bool>(levelCols,false));
        vector<int> iniXCol(levelCols);
        vector<int> iniYRow(levelRows);
        int nNoMore = 0;
        int nToDistribute = 0;


        float hY = cellH + 6;

        for(int i=0; i<levelRows; i++)
        {
            const float iniY = minBorderY + i*cellH - 3;
            iniYRow[i] = iniY;

            if(i == levelRows-1)
            {
                hY = maxBorderY+3-iniY;
                if(hY<=0)
                    continue;
            }

            float hX = cellW + 6;

            for(int j=0; j<levelCols; j++)
            {
                float iniX;

                if(i==0)
                {
                    iniX = minBorderX + j*cellW - 3;
                    iniXCol[j] = iniX;
                }
                else
                {
                    iniX = iniXCol[j];
                }


                if(j == levelCols-1)
                {
                    hX = maxBorderX+3-iniX;
                    if(hX<=0)
                        continue;
                }


                Mat cellImage = mvImagePyramid[level].rowRange(iniY,iniY+hY).colRange(iniX,iniX+hX);

                cellKeyPoints[i][j].reserve(nfeaturesCell*5);

                FAST(cellImage,cellKeyPoints[i][j],iniThFAST,true);

                if(cellKeyPoints[i][j].size()<=3)
                {
                    cellKeyPoints[i][j].clear();

                    FAST(cellImage,cellKeyPoints[i][j],minThFAST,true);
                }


                const int nKeys = cellKeyPoints[i][j].size();
                nTotal[i][j] = nKeys;

                if(nKeys>nfeaturesCell)
                {
                    nToRetain[i][j] = nfeaturesCell;
                    bNoMore[i][j] = false;
                }
                else
                {
                    nToRetain[i][j] = nKeys;
                    nToDistribute += nfeaturesCell-nKeys;
                    bNoMore[i][j] = true;
                    nNoMore++;
                }

            }
        }


        // Retain by score

        while(nToDistribute>0 && nNoMore<nCells)
        {
            int nNewFeaturesCell = nfeaturesCell + ceil((float)nToDistribute/(nCells-nNoMore));
            nToDistribute = 0;

            for(int i=0; i<levelRows; i++)
            {
                for(int j=0; j<levelCols; j++)
                {
                    if(!bNoMore[i][j])
                    {
                        if(nTotal[i][j]>nNewFeaturesCell)
                        {
                            nToRetain[i][j] = nNewFeaturesCell;
                            bNoMore[i][j] = false;
                        }
                        else
                        {
                            nToRetain[i][j] = nTotal[i][j];
                            nToDistribute += nNewFeaturesCell-nTotal[i][j];
                            bNoMore[i][j] = true;
                            nNoMore++;
                        }
                    }
                }
            }
        }

        vector<KeyPoint> & keypoints = allKeypoints[level];
        keypoints.reserve(nDesiredFeatures*2);

        const int scaledPatchSize = PATCH_SIZE*mvScaleFactor[level];

        // Retain by score and transform coordinates
        for(int i=0; i<levelRows; i++)
        {
            for(int j=0; j<levelCols; j++)
            {
                vector<KeyPoint> &keysCell = cellKeyPoints[i][j];
                KeyPointsFilter::retainBest(keysCell,nToRetain[i][j]);
                if((int)keysCell.size()>nToRetain[i][j])
                    keysCell.resize(nToRetain[i][j]);


                for(size_t k=0, kend=keysCell.size(); k<kend; k++)
                {
                    keysCell[k].pt.x+=iniXCol[j];
                    keysCell[k].pt.y+=iniYRow[i];
                    keysCell[k].octave=level;
                    keysCell[k].size = scaledPatchSize;
                    keypoints.push_back(keysCell[k]);
                }
            }
        }

        if((int)keypoints.size()>nDesiredFeatures)
        {
            KeyPointsFilter::retainBest(keypoints,nDesiredFeatures);
            keypoints.resize(nDesiredFeatures);
        }
    }

    // and compute orientations
    for (int level = 0; level < nlevels; ++level)
        computeOrientation(mvImagePyramid[level], allKeypoints[level], umax);
}

static void computeDescriptors(
    const Mat& image,               //某层金字塔图像
    vector<KeyPoint>& keypoints,    //特征点vector容器
    Mat& descriptors,               //描述子
    const vector<Point>& pattern    //预先定义好固定随机点集
)
{   
    //先清空
    descriptors = Mat::zeros((int)keypoints.size(), 32, CV_8UC1);

    //遍历每一个特征点
    for (size_t i = 0; i < keypoints.size(); i++)
        //计算这个特征点的描述子
        computeOrbDescriptor(
            keypoints[i],           //要计算描述子的特征点
            image,                  //该特征点所在的图像
            &pattern[0],            //随机点集的首地址
            descriptors.ptr((int)i) //提取出来的描述子的保存位置
        );
}



//重载()运算符的仿函数
/**
 * todo step1：判断图像是否为单通道的灰度图
 * todo step2：计算图像金字图
 * todo step3: 特征点的提取和分配
 * todo step4: 计算描述子
 * todo step5：对图像进行高斯模糊，计算高斯模糊后图像特征点的描述子
 * todo step6：对非0层图像中的特征点的坐标恢复到第0层图像的坐标系下
 */
void ORBextractor::operator()( 
    InputArray _image,              //输入的图像
    InputArray _mask,               //用于辅助进行图像处理的掩膜
    vector<KeyPoint>& _keypoints,   //存放特征点的vector容器
    OutputArray _descriptors        //描述子mat
    )
{ 
    if(_image.empty())
        return;

    // *step1：判断图像是否为单通道的灰度图
    Mat image = _image.getMat();
    assert(image.type() == CV_8UC1 );



    //  *step2：计算图像金字图
    // 利用前面计算的缩放因子，循环计算每一层图像的大小，并将图像都居中
    // Pre-compute the scale pyramid
    ComputePyramid(image);


    /**
     * *step3: 特征点的提取和分配
    */
    //存储每层图像的关键点
    vector < vector<KeyPoint> > allKeypoints;
    //使用4叉树的方式计算每层图像的特征点并进行分配
    ComputeKeyPointsOctTree(allKeypoints);
    //ComputeKeyPointsOld(allKeypoints);


    /**
     * *step4: 计算描述子
    */
    Mat descriptors;

    //统计图像金字塔剔除后关键点的个数
    int nkeypoints = 0;
    for (int level = 0; level < nlevels; ++level)
        nkeypoints += (int)allKeypoints[level].size();
    //如果本图金字塔像没有特征点
    if( nkeypoints == 0 ) 
        //通过Mat类的release方法，强制清空矩阵的引用次数，释放矩阵数据
        _descriptors.release();
    //如果本图金字塔像有特征点
    else
    {   
        //创建描述子矩阵，这个矩阵时储存整个图像金字塔中的特征点的描述子
        _descriptors.create(
            nkeypoints, //矩阵的行数，对应特征点的总个数
            32,         //矩阵的列数，对应为使用32*8的256位描述子
            CV_8U       //矩阵元素的格式
        );
        descriptors = _descriptors.getMat(); 
    }

    //清空 用作返回特征点提取结果的 vector容器
    _keypoints.clear();
    //并分配正确大小的空间
    _keypoints.reserve(nkeypoints);

    //因为遍历是一层图像一层图像进行的，当时描述子矩阵存储的是所有的特征点的描述子，所以这里设置了offset变量来保存“寻址”时的偏移量
    //辅助进行在总mat中进行定位
    int offset = 0;
    //开始遍历每一层图像
    for (int level = 0; level < nlevels; ++level)
    {   
        //当前层特征点容器的句柄
        vector<KeyPoint>& keypoints = allKeypoints[level];
        //本层的特征点数
        int nkeypointsLevel = (int)keypoints.size();
        //如果为0，继续计算下一层
        if(nkeypointsLevel==0)
            continue;

       
       
        // *step5：对图像进行高斯模糊
        //深拷贝当前层的图像
        Mat workingMat = mvImagePyramid[level].clone();
        /**
         * OpenCV的库函数GaussianBlur
         * 参数一：原图像
         * 参数二：目标图像
         * 参数三：高斯滤波器Kernel大小，必须为正的奇数
         * 参数四：高斯滤波在x轴方向的标准差
         * 参数五：高斯滤波在y轴方向的标准差
         * 参数六：边缘扩展点插值类型
        */
        GaussianBlur(workingMat, workingMat, Size(7, 7), 2, 2, BORDER_REFLECT_101);

        
        
        
        //desc 存储当前图层的描述子
        Mat desc = descriptors.rowRange(offset, offset + nkeypointsLevel);
        // *计算高斯模糊后图像特征点的描述子
        computeDescriptors(
            workingMat,     //高斯模糊之后的图层图像
            keypoints,      //当前图层的特征点集合
            desc,           //存储计算之后的描述子
            pattern         //随机采样点集，预先定义好的
        );

        //更新偏移量的值
        offset += nkeypointsLevel;


        /**
         * *step6：对非0层图像中的特征点的坐标恢复到第0层图像的坐标系下
         * *得到所有层特征点在第0层里的坐标放到_keypoints里面
         * */ 
        if (level != 0)
        {   
            //获取当前层的缩放系数
            float scale = mvScaleFactor[level]; //getScale(level, firstLevel, scaleFactor);
            //遍历本层所有的特征点
            for (vector<KeyPoint>::iterator keypoint = keypoints.begin(),
                 keypointEnd = keypoints.end(); keypoint != keypointEnd; ++keypoint)
                //特征点直接乘以缩放系数即可
                keypoint->pt *= scale;
        }
        // And add the keypoints to the output
        //将keypoints的内容插入到_keypoints的末尾
        //keypoints其实是对allKeypoints每层图像的引用
        //这样allKeypoints中的所有特征点在这里被转存到输出参数_keypoints中
        _keypoints.insert(_keypoints.end(), keypoints.begin(), keypoints.end());
    }
}


/**
 * *计算图像金字塔
*/
void ORBextractor::ComputePyramid(cv::Mat image)
{   
    //遍历所有图层
    for (int level = 0; level < nlevels; ++level)
    {   
        //获取本层图像的缩放系数
        float scale = mvInvScaleFactor[level];
        //计算本层图像的像素尺寸大小
        Size sz(cvRound((float)image.cols*scale), cvRound((float)image.rows*scale));

        //在已计算的本层图像的像素尺寸再向外扩展 长和高都延长EDGE_THRESHOLD*2的长度
        Size wholeSize(sz.width + EDGE_THRESHOLD*2, sz.height + EDGE_THRESHOLD*2);

        
        //声明两个临时变量，第一个和扩展图像相同大小；第二个作为掩膜，但后面未使用到它
        Mat temp(wholeSize, image.type()), masktemp;

        //temp图像中间的原图浅copy给图像金字塔该图层的图像
        /**
         * Rect(int x, int y, int width, int height);
         * Rect（左上角x坐标 ， 左上角y坐标，矩形的宽，矩形的高）
         * Rect函数讲解：https://blog.csdn.net/sinat_37281674/article/details/119478646
        */
        mvImagePyramid[level] = temp(Rect(EDGE_THRESHOLD, EDGE_THRESHOLD, sz.width, sz.height));

        // Compute the resized image
        //计算非0楼层图像的resized
        if( level != 0 )
        {   
            /**
             * void cv::resize(
             *      InputArray    src,                          原图像
             *      OutputArray   dst,                          输出图像
             *      Size          dsize                         输出图像的大小
             *      double        fx = 0,                       沿水平轴的缩放系数,0表示自动计算
             *      double        fy = 0,                       沿垂直轴的缩放系数，0表示自动计算
             *      int           interpolation = INTER_LINEAR  用于指定插值方式，默认为INTER_LINEAR（线性插值）
               )		
            */
            resize(mvImagePyramid[level-1], mvImagePyramid[level], sz, 0, 0, INTER_LINEAR);
            /**
             * copyMakeBorder 将源图像复制到目标图像的中间，并在图像周围填充像素形成边框，图片如果已经拷贝到中间，则只填充边界
             * EDGE_THRESHOLD指边界的宽度，当然不能在边界内提取特征点
            */
            copyMakeBorder(
                mvImagePyramid[level],              //原图像
                temp,                               //目标图像
                EDGE_THRESHOLD, EDGE_THRESHOLD,     //top & bottom需要扩展的border大小                     
                EDGE_THRESHOLD, EDGE_THRESHOLD,     //left & right需要扩展的border大小
                BORDER_REFLECT_101+BORDER_ISOLATED  //扩充方式 //?资料P21
            );            
        }
        //对于底层图像，直接扩充边界
        else    
        {   
            copyMakeBorder(
                image,          //原始图像
                temp,           //扩充图像
                EDGE_THRESHOLD, 
                EDGE_THRESHOLD, 
                EDGE_THRESHOLD, 
                EDGE_THRESHOLD,
                BORDER_REFLECT_101
            );            
        }
    }

}

} //namespace ORB_SLAM
