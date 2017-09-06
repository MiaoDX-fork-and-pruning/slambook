#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>
// #include "extra.h" // used in opencv2 

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <g2o/core/base_vertex.h>
#include <g2o/core/base_unary_edge.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/solvers/csparse/linear_solver_csparse.h>
#include <g2o/types/sba/types_six_dof_expmap.h>
#include <chrono>

#include <numeric>      // std::iota
#include <random>       // std::default_random_engine
#include <map> 
#include <math.h>

using namespace std;
using namespace cv;

void find_feature_matches (
    const Mat& img_1, const Mat& img_2,
    std::vector<KeyPoint>& keypoints_1,
    std::vector<KeyPoint>& keypoints_2,
    std::vector< DMatch >& matches );

void find_feature_matches_from_keypoints (
    const Mat& img_1, const Mat& img_2,
    std::vector<KeyPoint>& keypoints_1,
    std::vector<KeyPoint>& keypoints_2,
    std::vector< DMatch >& matches );

void pose_estimation_2d2d (
    const std::vector<KeyPoint>& keypoints_1,
    const std::vector<KeyPoint>& keypoints_2,
    const std::vector< DMatch >& matches,
	const Mat K,
    Mat& R, Mat& t );

void triangulation (
    const vector<KeyPoint>& keypoint_1,
    const vector<KeyPoint>& keypoint_2,
    const std::vector< DMatch >& matches,
	const Mat K,
    const Mat& R, const Mat& t,
    vector<Point3d>& points
);

void PNPSolver (
    const vector<KeyPoint>& keypoints_1,
    const vector<KeyPoint>& keypoints_2,
    const std::vector< DMatch >& matches,
    const Mat& d1,
	const Mat K
);

void corruptPoints (
    const vector<KeyPoint>& keypoint,
    const vector<Point3d>& points3d,
    vector<KeyPoint>& corruptKeypoint,
    vector<Point3d>& corruptPoints3d
);


void PNPSolver_img2_matched_and_3DPoints (
    const vector<Point2d>& points_img2,
    const vector<Point3d>& points_3d,
    const Mat& K );

void bundleAdjustment (
    const vector<Point3d> points_3d,
    const vector<Point2d> points_2d,
    const Mat& K,
    Mat& R, Mat& t
);

void DebugMatchedKeyPoints (
    const Mat& img_1, const Mat& img_2,
    const std::vector<KeyPoint>& keypoints_1,
    const std::vector<KeyPoint>& keypoints_2,
    const std::vector< DMatch >& matches
);


// [Copy from SFM](https://github.com/opencv/opencv_contrib/blob/master/modules/sfm/src/fundamental.cpp)
void essentialFromFundamental ( const Mat &F,
	const Mat &K1,
	const Mat &K2,
	Mat& E )
{
	E = K2.t () * F * K1;
}

// 像素坐标转相机归一化坐标
Point2f pixel2cam ( const Point2d& p, const Mat& K );

void rotate_angle ( Mat R )
{
	double r11 = R.at<double> ( 0, 0 ), r21 = R.at<double> ( 1, 0 ), r31 = R.at<double> ( 2, 0 ), r32 = R.at<double> ( 2, 1 ), r33 = R.at<double> ( 2, 2 );

	//计算出相机坐标系的三轴旋转欧拉角，旋转后可以转出世界坐标系。
	//旋转顺序为z、y、x

	const double PI = 3.14159265358979323846;
	double thetaz = atan2 ( r21, r11 ) / PI * 180;
	double thetay = atan2 ( -1 * r31, sqrt ( r32*r32 + r33*r33 ) ) / PI * 180;
	double thetax = atan2 ( r32, r33 ) / PI * 180;

	cout << "thetaz:" << thetaz << " thetay:" << thetay << " thetax:" << thetax << endl;
}


int main ( int argc, char** argv )
{
    if ( argc != 4 )
    {
        cout << "usage: triangulation img1 img2 depth1" << endl;
        return 1;
    }
    //-- 读取图像
    Mat img_1 = imread ( argv[1], CV_LOAD_IMAGE_COLOR );
    Mat img_2 = imread ( argv[2], CV_LOAD_IMAGE_COLOR );

    //Mat K = (Mat_<double> ( 3, 3 ) << 520.9, 0, 325.1, 0, 521.0, 249.7, 0, 0, 1);
	//K = np.array ( [[8607.8639, 0, 2880.72115], [0, 8605.4303, 1913.87935], [0, 0, 1]] ) # Canon5DMarkIII - EF50mm
	Mat K = (Mat_<double> ( 3, 3 ) << 8607.8639, 0, 2880.72115, 0, 8605.4303, 1913.87935, 0, 0, 1);


    vector<KeyPoint> keypoints_1_all, keypoints_2_all;
    vector<DMatch> matches;
    find_feature_matches ( img_1, img_2, keypoints_1_all, keypoints_2_all, matches );
	cout << "We found " << keypoints_1_all.size () << " key points in im1" << endl;
	cout << "We found " << keypoints_2_all.size () << " key points in im2" << endl;
    cout << "We found " << matches.size () << " pairs of points in total" << endl;
    // DebugMatchedKeyPoints ( img_1, img_2, keypoints_1_all, keypoints_2_all, matches );


    //-- 估计两张图像间运动
    Mat R, t;
    pose_estimation_2d2d ( keypoints_1_all, keypoints_2_all, matches, K, R, t );
    /*
    // Resign the t vector, according to the physics world
    if (signbit(t.at<double>(0, 0))) { // negative
        t *= -10.0 / t.at<double>(0, 0); // We can assign the absolute valuse of the t, and then the calculated points will have absolute dimension
    }
    else
    {
        t *= 10.0 / t.at<double>(0, 0);
    }

    cout << "Covered t is " << endl << t << endl;
    */

    //-- 三角化
    vector<Point3d> points_3d_matched;
    triangulation ( keypoints_1_all, keypoints_2_all, matches, K, R, t, points_3d_matched );

    /*
    // 建立3D点
    Mat d1 = imread ( argv[3], CV_LOAD_IMAGE_UNCHANGED );       // 深度图为16位无符号数，单通道图像
    PNPSolver ( keypoints_1_all, keypoints_2_all, matches, d1, K );
	*/
    vector<Point2d> points_im2_matched;
    for ( DMatch m : matches ) {
        points_im2_matched.push_back ( keypoints_2_all[m.trainIdx].pt ); // no need to convert to camera coordinate
    }
    PNPSolver_img2_matched_and_3DPoints ( points_im2_matched, points_3d_matched, K );

    // Remove some of the chosen key points and 3d points pairs, to simulate manipulation on them
    vector<KeyPoint> im1_corruptKeypoint;
    vector<Point3d> corruptPoints3d;
    vector<KeyPoint> keypoint_1_matched;
    for ( DMatch m : matches ) {
        keypoint_1_matched.push_back ( keypoints_1_all[m.queryIdx] );
    }  
    corruptPoints ( keypoint_1_matched, points_3d_matched, im1_corruptKeypoint, corruptPoints3d );

    vector<DMatch> matchesAfterCorrupt;
    find_feature_matches_from_keypoints ( img_1, img_2, im1_corruptKeypoint, keypoints_2_all, matchesAfterCorrupt );
    DebugMatchedKeyPoints ( img_1, img_2, im1_corruptKeypoint, keypoints_2_all, matchesAfterCorrupt );
    
    vector<Point3d> points_3d_corrupted_and_mathced;
    vector<Point2d> points_im2_corrupted_and_mathced;
    for ( DMatch m : matchesAfterCorrupt )
    {
        points_3d_corrupted_and_mathced.push_back ( corruptPoints3d[m.queryIdx] );
        points_im2_corrupted_and_mathced.push_back ( keypoints_2_all[m.trainIdx].pt );
    }
    
    PNPSolver_img2_matched_and_3DPoints ( points_im2_corrupted_and_mathced, points_3d_corrupted_and_mathced, K );

    system ( "pause" );

    return 0;
}

void find_feature_matches ( const Mat& img_1, const Mat& img_2,
    std::vector<KeyPoint>& keypoints_1,
    std::vector<KeyPoint>& keypoints_2,
    std::vector< DMatch >& matches )
{
    //-- 初始化
    Mat descriptors_1, descriptors_2;
    // used in OpenCV3 
    Ptr<FeatureDetector> detector = ORB::create ();
    //-- 第一步:检测 Oriented FAST 角点位置
    detector->detect ( img_1, keypoints_1 );
    detector->detect ( img_2, keypoints_2 );

	/*
	for ( int i = 0; i < 10; i++ ) {
		Point2d p2 = keypoints_1[i].pt;
		cout << p2.x << " " << p2.y << endl;
	}
	*/

    find_feature_matches_from_keypoints ( img_1, img_2, keypoints_1, keypoints_2, matches );
}



void find_feature_matches_from_keypoints (
    const Mat& img_1, const Mat& img_2,
    std::vector<KeyPoint>& keypoints_1,
    std::vector<KeyPoint>& keypoints_2,
    std::vector< DMatch >& matches )
{
    //-- 初始化
    Mat descriptors_1, descriptors_2;
    // used in OpenCV3 
    Ptr<DescriptorExtractor> descriptor = ORB::create ();
    // use this if you are in OpenCV2 
    // Ptr<FeatureDetector> detector = FeatureDetector::create ( "ORB" );
    // Ptr<DescriptorExtractor> descriptor = DescriptorExtractor::create ( "ORB" );
    Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create ( "BruteForce-Hamming" );

    //-- 第二步:根据角点位置计算 BRIEF 描述子
    descriptor->compute ( img_1, keypoints_1, descriptors_1 );
    descriptor->compute ( img_2, keypoints_2, descriptors_2 );

    //-- 第三步:对两幅图像中的BRIEF描述子进行匹配，使用 Hamming 距离
    vector<DMatch> match;
    // BFMatcher matcher ( NORM_HAMMING );
    matcher->match ( descriptors_1, descriptors_2, match );

    //-- 第四步:匹配点对筛选
    double min_dist = 10000, max_dist = 0;

    //找出所有匹配之间的最小距离和最大距离, 即是最相似的和最不相似的两组点之间的距离
    for ( int i = 0; i < descriptors_1.rows; i++ )
    {
        double dist = match[i].distance;
        if ( dist < min_dist ) min_dist = dist;
        if ( dist > max_dist ) max_dist = dist;
    }

    printf ( "-- Max dist : %f \n", max_dist );
    printf ( "-- Min dist : %f \n", min_dist );

    //当描述子之间的距离大于两倍的最小距离时,即认为匹配有误.但有时候最小距离会非常小,设置一个经验值30作为下限.
    for ( int i = 0; i < descriptors_1.rows; i++ )
    {
        if ( match[i].distance <= max ( 2 * min_dist, 30.0 ) )
        {
            matches.push_back ( match[i] );
        }
    }
}


void DebugMatchedKeyPoints (
    const Mat& img_1, const Mat& img_2,
    const std::vector<KeyPoint>& keypoints_1,
    const std::vector<KeyPoint>& keypoints_2,
    const std::vector< DMatch >& matches
)
{
    std::vector<KeyPoint> keypoints_1_matched;
    std::vector<KeyPoint> keypoints_2_matched;
    std::vector<Point2d> points_1_matched;
    std::vector<Point2d> points_2_matched;

    /* Debug */
    cout << "Output all matched keypoints and corresponding matches and their distance" << endl;
    for ( DMatch m : matches ) {
        keypoints_1_matched.push_back ( keypoints_1[m.queryIdx] );
        points_1_matched.push_back ( keypoints_1[m.queryIdx].pt );
        
        keypoints_2_matched.push_back ( keypoints_2[m.trainIdx] );
        points_2_matched.push_back ( keypoints_2[m.trainIdx].pt );
    }

    //-- Draw the descriptors
    Mat outimg1, outimg2;
    drawKeypoints ( img_1, keypoints_1, outimg1, Scalar::all ( -1 ), DrawMatchesFlags::DEFAULT );
    drawKeypoints ( img_2, keypoints_2, outimg2, Scalar::all ( -1 ), DrawMatchesFlags::DEFAULT );
    imshow ( "Descriptors on im1", outimg1 );
    imshow ( "Descriptors on im2", outimg2 );

    //-- 第五步:绘制匹配结果
    Mat img_match;
    Mat img_goodmatch;
    drawMatches ( img_1, keypoints_1, img_2, keypoints_2, matches, img_match );
    printf ( "-- All match num : %I64u \n", matches.size () );
    imshow ( "All matched points", img_match );
    waitKey ( 0 );

}


void pose_estimation_2d2d (
    const std::vector<KeyPoint>& keypoints_1,
    const std::vector<KeyPoint>& keypoints_2,
    const std::vector< DMatch >& matches,
	const Mat K,
    Mat& R, Mat& t )
{
    // 相机内参,TUM Freiburg2
    //Mat K = (Mat_<double> ( 3, 3 ) << 520.9, 0, 325.1, 0, 521.0, 249.7, 0, 0, 1);

    //-- 把匹配点转换为vector<Point2f>的形式
	vector<Point2f> points1;
	vector<Point2f> points2;


	for ( auto m : matches )
	{
		points1.push_back ( keypoints_1[m.queryIdx].pt );
		points2.push_back ( keypoints_2[m.trainIdx].pt );
	}

	/*
	for ( int i = 0; i < points1.size(); i++ ) {
		Point2d p1 = points1[i];
		Point2d p2 = points2[i];
		cout << "i:" << i << endl;
		cout << "p1:" << p1.x << " " << p1.y << endl;
		cout << "p2:" << p2.x << " " << p2.y << endl;
	}
	*/

    //-- 计算基础矩阵
    Mat fundamental_matrix;
    //fundamental_matrix = findFundamentalMat ( points1, points2, CV_FM_8POINT );
	fundamental_matrix = findFundamentalMat ( points1, points2, CV_RANSAC, 0.1, 0.99 );
    cout << "fundamental_matrix is " << endl << fundamental_matrix << endl;

	/*
    //-- 计算本质矩阵
    Point2d principal_point ( 325.1, 249.7 );				//相机主点, TUM dataset标定值
    int focal_length = 521;						//相机焦距, TUM dataset标定值
    Mat essential_matrix;
    essential_matrix = findEssentialMat ( points1, points2, focal_length, principal_point );
    cout << "essential_matrix is " << endl << essential_matrix << endl;
	*/
	Mat essential_matrix = findEssentialMat ( points1, points2, K );
	cout << "essential_matrix2 is " << endl << essential_matrix << endl;

	
	// -- F -> E
	Mat essential_matrix3;
	essentialFromFundamental ( fundamental_matrix, K, K, essential_matrix3 );
	cout << "essential_matrix3 is " << endl << essential_matrix3 << endl;

    //-- 计算单应矩阵
    Mat homography_matrix;
    homography_matrix = findHomography ( points1, points2, RANSAC, 3 );
    cout << "homography_matrix is " << endl << homography_matrix << endl;

    //-- 从本质矩阵中恢复旋转和平移信息.

	//recoverPose ( essential_matrix, points1, points2, R, t, focal_length, principal_point );
	recoverPose ( essential_matrix, points1, points2, K, R, t);
	Mat r;
	cv::Rodrigues ( R, r ); // r为旋转向量形式，用Rodrigues公式转换为矩阵

	cout << "R=" << endl << R << endl;
	cout << "r=" << endl << r << endl;
	rotate_angle ( R );
    cout << "t is " << endl << t << endl;
}

void triangulation (	
    const vector< KeyPoint >& keypoint_1,
    const vector< KeyPoint >& keypoint_2,
    const std::vector< DMatch >& matches,
	const Mat K,
    const Mat& R, const Mat& t,
    vector< Point3d >& points )
{
    Mat T1 = (Mat_<float> ( 3, 4 ) <<
        1, 0, 0, 0,
        0, 1, 0, 0,
        0, 0, 1, 0);
    Mat T2 = (Mat_<float> ( 3, 4 ) <<
        R.at<double> ( 0, 0 ), R.at<double> ( 0, 1 ), R.at<double> ( 0, 2 ), t.at<double> ( 0, 0 ),
        R.at<double> ( 1, 0 ), R.at<double> ( 1, 1 ), R.at<double> ( 1, 2 ), t.at<double> ( 1, 0 ),
        R.at<double> ( 2, 0 ), R.at<double> ( 2, 1 ), R.at<double> ( 2, 2 ), t.at<double> ( 2, 0 )
        );

    //Mat K = (Mat_<double> ( 3, 3 ) << 520.9, 0, 325.1, 0, 521.0, 249.7, 0, 0, 1);
    vector<Point2f> pts_1, pts_2;
    for ( DMatch m : matches )
    {
        // 将像素坐标转换至相机坐标
        pts_1.push_back ( pixel2cam ( keypoint_1[m.queryIdx].pt, K ) );
        pts_2.push_back ( pixel2cam ( keypoint_2[m.trainIdx].pt, K ) );
    }

    Mat pts_4d;
    cv::triangulatePoints ( T1, T2, pts_1, pts_2, pts_4d );

    // 转换成非齐次坐标
    for ( int i = 0; i < pts_4d.cols; i++ )
    {
        Mat x = pts_4d.col ( i );
        x /= x.at<float> ( 3, 0 ); // 归一化
        Point3d p (
            x.at<float> ( 0, 0 ),
            x.at<float> ( 1, 0 ),
            x.at<float> ( 2, 0 )
        );
        points.push_back ( p );
    }

    /*
    //-- 验证三角化点与特征点的重投影关系
    // for ( int i=0; i<matches.size(); i++ )
    for (int i = 0; i< 2; i++)
    {
    Point2d pt1_cam = pixel2cam( keypoints_1[ matches[i].queryIdx ].pt, K );
    Point2d pt1_cam_3d(
    points[i].x/points[i].z,
    points[i].y/points[i].z
    );

    cout<<"point in the first camera frame: "<<pt1_cam<<endl;
    cout<<"point projected from 3D "<<pt1_cam_3d<<", d="<<points[i].z<<endl;

    // 第二个图
    Point2f pt2_cam = pixel2cam( keypoints_2[ matches[i].trainIdx ].pt, K );
    Mat pt2_trans = R*( Mat_<double>(3,1) << points[i].x, points[i].y, points[i].z ) + t;
    pt2_trans /= pt2_trans.at<double>(2,0);
    cout<<"point in the second camera frame: "<<pt2_cam<<endl;
    cout<<"point reprojected from second frame: "<<pt2_trans.t()<<endl;
    cout<<endl;
    }
    */

}

Point2f pixel2cam ( const Point2d& p, const Mat& K )
{
	//[1、像素坐标与像平面坐标系之间的关系 ](http://blog.csdn.net/waeceo/article/details/50580607)
    return Point2f
    (
        (p.x - K.at<double> ( 0, 2 )) / K.at<double> ( 0, 0 ),
        (p.y - K.at<double> ( 1, 2 )) / K.at<double> ( 1, 1 )
    );
}



void PNPSolver (
    const vector<KeyPoint>& keypoints_1,
    const vector<KeyPoint>& keypoints_2,
    const std::vector< DMatch >& matches,
    const Mat& d1,
	const Mat K
)
{
    //Mat K = (Mat_<double> ( 3, 3 ) << 520.9, 0, 325.1, 0, 521.0, 249.7, 0, 0, 1);
    vector<Point3d> pts_3d;
    vector<Point2d> pts_2d;
    for ( DMatch m : matches )
    {
        ushort d = d1.ptr<unsigned short> ( int ( keypoints_1[m.queryIdx].pt.y ) )[int ( keypoints_1[m.queryIdx].pt.x )];
        if ( d == 0 )   // bad depth
            continue;
        float dd = d / 1000.0;
        Point2d p1 = pixel2cam ( keypoints_1[m.queryIdx].pt, K );
        pts_3d.push_back ( Point3f ( p1.x*dd, p1.y*dd, dd ) );
        pts_2d.push_back ( keypoints_2[m.trainIdx].pt );
    }
    /*
    cout << "pts_3d" << endl;
    for (auto p : pts_3d) {
        cout << p.x << " " << p.y << " " << p.z << endl;
    }
    */

    cout << "3d-2d pairs: " << pts_3d.size () << endl;

    Mat r, t;
    solvePnP ( pts_3d, pts_2d, K, Mat (), r, t, false ); // 调用OpenCV 的 PnP 求解，可选择EPNP，DLS等方法
    Mat R;
    cv::Rodrigues ( r, R ); // r为旋转向量形式，用Rodrigues公式转换为矩阵

    cout << "R=" << endl << R << endl;
    cout << "r=" << endl << r << endl;
    cout << "t=" << endl << t << endl;

    cout << "calling bundle adjustment" << endl;

    bundleAdjustment ( pts_3d, pts_2d, K, R, t );

}




void PNPSolver_img2_matched_and_3DPoints (
    const vector<Point2d>& points_img2,
    const vector<Point3d>& points_3d,
    const Mat& K
)
{
    // Mat K = (Mat_<double> ( 3, 3 ) << 520.9, 0, 325.1, 0, 521.0, 249.7, 0, 0, 1);

    assert ( points_img2.size () == points_3d.size () );

    cout << "3d-2d pairs: " << points_3d.size () << endl;

    Mat r, t;
    solvePnP ( points_3d, points_img2, K, Mat (), r, t, false );

    Mat R;
    cv::Rodrigues ( r, R ); // r为旋转向量形式，用Rodrigues公式转换为矩阵

    cout << "R=" << endl << R << endl;
    cout << "r=" << endl << r << endl;
    cout << "t=" << endl << t << endl;

    cout << "calling bundle adjustment" << endl;

    bundleAdjustment ( points_3d, points_img2, K, R, t );
}


void bundleAdjustment (
    const vector< Point3d > points_3d,
    const vector< Point2d > points_2d,
    const Mat& K,
    Mat& R, Mat& t )
{
    // 初始化g2o
    typedef g2o::BlockSolver< g2o::BlockSolverTraits<6, 3> > Block;  // pose 维度为 6, landmark 维度为 3
    Block::LinearSolverType* linearSolver = new g2o::LinearSolverCSparse<Block::PoseMatrixType> (); // 线性方程求解器
    Block* solver_ptr = new Block ( linearSolver );     // 矩阵块求解器
    g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg ( solver_ptr );
    g2o::SparseOptimizer optimizer;
    optimizer.setAlgorithm ( solver );

    // vertex
    g2o::VertexSE3Expmap* pose = new g2o::VertexSE3Expmap (); // camera pose
    Eigen::Matrix3d R_mat;
    R_mat <<
        R.at<double> ( 0, 0 ), R.at<double> ( 0, 1 ), R.at<double> ( 0, 2 ),
        R.at<double> ( 1, 0 ), R.at<double> ( 1, 1 ), R.at<double> ( 1, 2 ),
        R.at<double> ( 2, 0 ), R.at<double> ( 2, 1 ), R.at<double> ( 2, 2 );
    pose->setId ( 0 );
    pose->setEstimate ( g2o::SE3Quat (
        R_mat,
        Eigen::Vector3d ( t.at<double> ( 0, 0 ), t.at<double> ( 1, 0 ), t.at<double> ( 2, 0 ) )
    ) );
    optimizer.addVertex ( pose );

    int index = 1;
    for ( const Point3f p : points_3d )   // landmarks
    {
        g2o::VertexSBAPointXYZ* point = new g2o::VertexSBAPointXYZ ();
        point->setId ( index++ );
        point->setEstimate ( Eigen::Vector3d ( p.x, p.y, p.z ) );
        point->setMarginalized ( true ); // g2o 中必须设置 marg 参见第十讲内容
        optimizer.addVertex ( point );
    }

    // parameter: camera intrinsics
    g2o::CameraParameters* camera = new g2o::CameraParameters (
        K.at<double> ( 0, 0 ), Eigen::Vector2d ( K.at<double> ( 0, 2 ), K.at<double> ( 1, 2 ) ), 0
    );
    camera->setId ( 0 );
    optimizer.addParameter ( camera );

    // edges
    index = 1;
    for ( const Point2f p : points_2d )
    {
        g2o::EdgeProjectXYZ2UV* edge = new g2o::EdgeProjectXYZ2UV ();
        edge->setId ( index );
        edge->setVertex ( 0, dynamic_cast<g2o::VertexSBAPointXYZ*> (optimizer.vertex ( index )) );
        edge->setVertex ( 1, pose );
        edge->setMeasurement ( Eigen::Vector2d ( p.x, p.y ) );
        edge->setParameterId ( 0, 0 );
        edge->setInformation ( Eigen::Matrix2d::Identity () );
        optimizer.addEdge ( edge );
        index++;
    }

    chrono::steady_clock::time_point t1 = chrono::steady_clock::now ();
    optimizer.setVerbose ( true );
    optimizer.initializeOptimization ();
    optimizer.optimize ( 100 );
    chrono::steady_clock::time_point t2 = chrono::steady_clock::now ();
    chrono::duration<double> time_used = chrono::duration_cast<chrono::duration<double>> (t2 - t1);
    cout << "optimization costs time: " << time_used.count () << " seconds." << endl;

    cout << endl << "after optimization:" << endl;
    cout << "T=" << endl << Eigen::Isometry3d ( pose->estimate () ).matrix () << endl;

}

void corruptPoints (
    const vector<KeyPoint>& keypoint,
    const vector<Point3d>& points3d,
    vector<KeyPoint>& corruptKeypoint,
    vector<Point3d>& corruptPoints3d
)
{
    assert ( keypoint.size () == points3d.size () ); // assert will do nothing for release

    float corrupt_ratio = 0.5;

    std::vector<int> indexes ( keypoint.size(), 0 );
    std::iota ( indexes.begin (), indexes.end (), 0 );
    shuffle ( indexes.begin (), indexes.end (), std::default_random_engine ( 42 ) );

    for ( int i = 0; i < (int)(points3d.size () * corrupt_ratio); i++ ) {
        corruptKeypoint.push_back ( keypoint[i] );
        corruptPoints3d.push_back ( points3d[i] );
    }
}