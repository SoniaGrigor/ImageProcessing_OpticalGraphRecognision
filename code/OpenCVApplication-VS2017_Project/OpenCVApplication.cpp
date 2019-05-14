// OpenCVApplication.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include "common.h"
#include <vector>
#include <iostream>
#include <stdio.h>
#include<random>
#include <algorithm>
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <opencv/cv.h>
#include <opencv/highgui.h>

#define DIGIT_HIGHT_MIN 15
#define DIGIT_HIGHT_MAX 50

#define DIGIT_WIDTH_MIN 15
#define DIGIT_WIDTH_MAX 50

using namespace cv;
using namespace std;

int di[4] = { 0, -1, -1, -1 };
int dj[4] = { -1, -1, 0, +1, };

Vec3b culoriRandom[1000];
int label;
int newLabel;

//vector<Vec3f> circles;
//vector<Vec4i> lines;

void testOpenImage()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src;
		src = imread(fname);
		imshow("image", src);
		waitKey();
	}
}

void testOpenImagesFld()
{
	char folderName[MAX_PATH];
	if (openFolderDlg(folderName) == 0)
		return;
	char fname[MAX_PATH];
	FileGetter fg(folderName, "bmp");
	while (fg.getNextAbsFile(fname))
	{
		Mat src;
		src = imread(fname);
		imshow(fg.getFoundFileName(), src);
		if (waitKey() == 27) //ESC pressed
			break;
	}
}

void testImageOpenAndSave()
{
	Mat src, dst;

	src = imread("Images/Lena_24bits.bmp", CV_LOAD_IMAGE_COLOR);	// Read the image

	if (!src.data)	// Check for invalid input
	{
		printf("Could not open or find the image\n");
		return;
	}

	// Get the image resolution
	Size src_size = Size(src.cols, src.rows);

	// Display window
	const char* WIN_SRC = "Src"; //window for the source image
	namedWindow(WIN_SRC, CV_WINDOW_AUTOSIZE);
	cvMoveWindow(WIN_SRC, 0, 0);

	const char* WIN_DST = "Dst"; //window for the destination (processed) image
	namedWindow(WIN_DST, CV_WINDOW_AUTOSIZE);
	cvMoveWindow(WIN_DST, src_size.width + 10, 0);

	cvtColor(src, dst, CV_BGR2GRAY); //converts the source image to a grayscale one

	imwrite("Images/Lena_24bits_gray.bmp", dst); //writes the destination to file

	imshow(WIN_SRC, src);
	imshow(WIN_DST, dst);

	printf("Press any key to continue ...\n");
	waitKey(0);
}

void testNegativeImage()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		double t = (double)getTickCount(); // Get the current time [s]

		Mat src = imread(fname, CV_LOAD_IMAGE_GRAYSCALE);
		int height = src.rows;
		int width = src.cols;
		Mat dst = Mat(height, width, CV_8UC1);
		// Asa se acceseaaza pixelii individuali pt. o imagine cu 8 biti/pixel
		// Varianta ineficienta (lenta)
		for (int i = 0; i < height; i++)
		{
			for (int j = 0; j < width; j++)
			{
				uchar val = src.at<uchar>(i, j);
				uchar neg = 255 - val;
				dst.at<uchar>(i, j) = neg;
			}
		}

		// Get the current time again and compute the time difference [s]
		t = ((double)getTickCount() - t) / getTickFrequency();
		// Print (in the console window) the processing time in [ms] 
		printf("Time = %.3f [ms]\n", t * 1000);

		imshow("input image", src);
		imshow("negative image", dst);
		waitKey();
	}
}

void testParcurgereSimplaDiblookStyle()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src = imread(fname, CV_LOAD_IMAGE_GRAYSCALE);
		int height = src.rows;
		int width = src.cols;
		Mat dst = src.clone();

		double t = (double)getTickCount(); // Get the current time [s]

										   // the fastest approach using the “diblook style”
		uchar *lpSrc = src.data;
		uchar *lpDst = dst.data;
		int w = (int)src.step; // no dword alignment is done !!!
		for (int i = 0; i < height; i++)
			for (int j = 0; j < width; j++) {
				uchar val = lpSrc[i*w + j];
				lpDst[i*w + j] = 255 - val;
			}

		// Get the current time again and compute the time difference [s]
		t = ((double)getTickCount() - t) / getTickFrequency();
		// Print (in the console window) the processing time in [ms] 
		printf("Time = %.3f [ms]\n", t * 1000);

		imshow("input image", src);
		imshow("negative image", dst);
		waitKey();
	}
}

void testColor2Gray()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src = imread(fname);

		int height = src.rows;
		int width = src.cols;

		Mat dst = Mat(height, width, CV_8UC1);

		// Asa se acceseaaza pixelii individuali pt. o imagine RGB 24 biti/pixel
		// Varianta ineficienta (lenta)
		for (int i = 0; i < height; i++)
		{
			for (int j = 0; j < width; j++)
			{
				Vec3b v3 = src.at<Vec3b>(i, j);
				uchar b = v3[0];
				uchar g = v3[1];
				uchar r = v3[2];
				dst.at<uchar>(i, j) = (r + g + b) / 3;
			}
		}

		imshow("input image", src);
		imshow("gray image", dst);
		waitKey();
	}
}

void testBGR2HSV()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src = imread(fname);
		int height = src.rows;
		int width = src.cols;

		// Componentele d eculoare ale modelului HSV
		Mat H = Mat(height, width, CV_8UC1);
		Mat S = Mat(height, width, CV_8UC1);
		Mat V = Mat(height, width, CV_8UC1);

		// definire pointeri la matricele (8 biti/pixeli) folosite la afisarea componentelor individuale H,S,V
		uchar* lpH = H.data;
		uchar* lpS = S.data;
		uchar* lpV = V.data;

		Mat hsvImg;
		cvtColor(src, hsvImg, CV_BGR2HSV);

		// definire pointer la matricea (24 biti/pixeli) a imaginii HSV
		uchar* hsvDataPtr = hsvImg.data;

		for (int i = 0; i < height; i++)
		{
			for (int j = 0; j < width; j++)
			{
				int hi = i * width * 3 + j * 3;
				int gi = i * width + j;

				lpH[gi] = hsvDataPtr[hi] * 510 / 360;		// lpH = 0 .. 255
				lpS[gi] = hsvDataPtr[hi + 1];			// lpS = 0 .. 255
				lpV[gi] = hsvDataPtr[hi + 2];			// lpV = 0 .. 255
			}
		}

		imshow("input image", src);
		imshow("H", H);
		imshow("S", S);
		imshow("V", V);

		waitKey();
	}
}

void testResize()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src;
		src = imread(fname);
		Mat dst1, dst2;
		//without interpolation
		resizeImg(src, dst1, 320, false);
		//with interpolation
		resizeImg(src, dst2, 320, true);
		imshow("input image", src);
		imshow("resized image (without interpolation)", dst1);
		imshow("resized image (with interpolation)", dst2);
		waitKey();
	}
}

void testCanny()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src, dst, gauss;
		src = imread(fname, CV_LOAD_IMAGE_GRAYSCALE);
		double k = 0.4;
		int pH = 50;
		int pL = (int)k*pH;
		GaussianBlur(src, gauss, Size(5, 5), 0.8, 0.8);
		Canny(gauss, dst, pL, pH, 3);
		imshow("input image", src);
		imshow("canny", dst);
		waitKey();
	}
}

void testVideoSequence()
{
	VideoCapture cap("Videos/rubic.avi"); // off-line video from file
										  //VideoCapture cap(0);	// live video from web cam
	if (!cap.isOpened()) {
		printf("Cannot open video capture device.\n");
		waitKey(0);
		return;
	}

	Mat edges;
	Mat frame;
	char c;

	while (cap.read(frame))
	{
		Mat grayFrame;
		cvtColor(frame, grayFrame, CV_BGR2GRAY);
		Canny(grayFrame, edges, 40, 100, 3);
		imshow("source", frame);
		imshow("gray", grayFrame);
		imshow("edges", edges);
		c = cvWaitKey(0);  // waits a key press to advance to the next frame
		if (c == 27) {
			// press ESC to exit
			printf("ESC pressed - capture finished\n");
			break;  //ESC pressed
		};
	}
}

void testSnap()
{
	VideoCapture cap(0); // open the deafult camera (i.e. the built in web cam)
	if (!cap.isOpened()) // openenig the video device failed
	{
		printf("Cannot open video capture device.\n");
		return;
	}

	Mat frame;
	char numberStr[256];
	char fileName[256];

	// video resolution
	Size capS = Size((int)cap.get(CV_CAP_PROP_FRAME_WIDTH),
		(int)cap.get(CV_CAP_PROP_FRAME_HEIGHT));

	// Display window
	const char* WIN_SRC = "Src"; //window for the source frame
	namedWindow(WIN_SRC, CV_WINDOW_AUTOSIZE);
	cvMoveWindow(WIN_SRC, 0, 0);

	const char* WIN_DST = "Snapped"; //window for showing the snapped frame
	namedWindow(WIN_DST, CV_WINDOW_AUTOSIZE);
	cvMoveWindow(WIN_DST, capS.width + 10, 0);

	char c;
	int frameNum = -1;
	int frameCount = 0;

	for (;;)
	{
		cap >> frame; // get a new frame from camera
		if (frame.empty())
		{
			printf("End of the video file\n");
			break;
		}

		++frameNum;

		imshow(WIN_SRC, frame);

		c = cvWaitKey(10);  // waits a key press to advance to the next frame
		if (c == 27) {
			// press ESC to exit
			printf("ESC pressed - capture finished");
			break;  //ESC pressed
		}
		if (c == 115) { //'s' pressed - snapp the image to a file
			frameCount++;
			fileName[0] = NULL;
			sprintf(numberStr, "%d", frameCount);
			strcat(fileName, "Images/A");
			strcat(fileName, numberStr);
			strcat(fileName, ".bmp");
			bool bSuccess = imwrite(fileName, frame);
			if (!bSuccess)
			{
				printf("Error writing the snapped image\n");
			}
			else
				imshow(WIN_DST, frame);
		}
	}

}

void MyCallBackFunc(int event, int x, int y, int flags, void* param)
{
	//More examples: http://opencvexamples.blogspot.com/2014/01/detect-mouse-clicks-and-moves-on-image.html
	Mat* src = (Mat*)param;
	if (event == CV_EVENT_LBUTTONDOWN)
	{
		printf("Pos(x,y): %d,%d  Color(RGB): %d,%d,%d\n",
			x, y,
			(int)(*src).at<Vec3b>(y, x)[2],
			(int)(*src).at<Vec3b>(y, x)[1],
			(int)(*src).at<Vec3b>(y, x)[0]);
	}
}

void testMouseClick()
{
	Mat src;
	// Read image from file 
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		src = imread(fname);
		//Create a window
		namedWindow("My Window", 1);

		//set the callback function for any mouse event
		setMouseCallback("My Window", MyCallBackFunc, &src);

		//show the image
		imshow("My Window", src);

		// Wait until user press some key
		waitKey(0);
	}
}

/* Histogram display function - display a histogram using bars (simlilar to L3 / PI)
Input:
name - destination (output) window name
hist - pointer to the vector containing the histogram values
hist_cols - no. of bins (elements) in the histogram = histogram image width
hist_height - height of the histogram image
Call example:
showHistogram ("MyHist", hist_dir, 255, 200);
*/
void showHistogram(const std::string& name, int* hist, const int  hist_cols, const int hist_height)
{
	Mat imgHist(hist_height, hist_cols, CV_8UC3, CV_RGB(255, 255, 255)); // constructs a white image

																		 //computes histogram maximum
	int max_hist = 0;
	for (int i = 0; i < hist_cols; i++)
		if (hist[i] > max_hist)
			max_hist = hist[i];
	double scale = 1.0;
	scale = (double)hist_height / max_hist;
	int baseline = hist_height - 1;

	for (int x = 0; x < hist_cols; x++) {
		Point p1 = Point(x, baseline);
		Point p2 = Point(x, baseline - cvRound(hist[x] * scale));
		line(imgHist, p1, p2, CV_RGB(255, 0, 255)); // histogram bins colored in magenta
	}

	imshow(name, imgHist);
}

int binaryPixel(int treshold, int initialValue) {
	if (initialValue >= treshold)
		return 255;
	if (initialValue < treshold)
		return 0;
}

Mat binaryImage(Mat src) {
	int height = src.rows;
	int width = src.cols;
	Mat dst = Mat(height, width, CV_8UC1);

	for (int i = 0; i < height; i++)
	{
		for (int j = 0; j < width; j++)
		{
			dst.at<uchar>(i, j) = binaryPixel(150, src.at<uchar>(i, j));
		}
	}
	return dst;
}

vector<Vec3f> houghCirclesFuncion(Mat src) {
	Mat src_gray = src.clone();
	vector<Vec3f> circles;
	/// Reduce the noise so we avoid false circle detection
	GaussianBlur(src, src, Size(9, 9), 2, 2);
	/// Apply the Hough Transform to find the circles
	HoughCircles(src, circles, CV_HOUGH_GRADIENT, 1, src.rows / 10, 100, 100, 10, 70);
	return circles;
}

Mat houghCirclesFuncionImage(Mat img) {
	Mat circlesImage(img.rows, img.cols, CV_8UC3);
	circlesImage.setTo(cv::Scalar(255, 255, 255));
	vector<Vec3f> circles = houghCirclesFuncion(img);
	/// Draw the circles detected
	for (size_t i = 0; i < circles.size(); i++)
	{
		Point center(cvRound(circles[i][0]), cvRound(circles[i][1]));
		int radius = cvRound(circles[i][2]);
		// circle center
		circle(circlesImage, center, 3, Scalar(0, 255, 255), -1, 8, 0);
		// circle outline
		circle(circlesImage, center, radius, Scalar(250, 0, 255), 3, 8, 0);
	}
	return circlesImage;
}

Mat detectCircles(Mat src) {
	Mat src_gray = src.clone();
	Mat result(src.rows, src.cols, CV_8UC3);
	result.setTo(cv::Scalar(255, 255, 255));
	vector<Vec3f> circles;

	/// Reduce the noise so we avoid false circle detection
	GaussianBlur(src, src, Size(9, 9), 2, 2);

	/// Apply the Hough Transform to find the circles
	HoughCircles(src, circles, CV_HOUGH_GRADIENT, 1, src.rows / 8, 200, 100, 0, 0);

	/// Draw the circles detected
	for (size_t i = 0; i < circles.size(); i++)
	{
		Point center(cvRound(circles[i][0]), cvRound(circles[i][1]));
		int radius = cvRound(circles[i][2]);
		// circle center
		circle(result, center, 3, Scalar(0, 255, 0), -1, 8, 0);
		// circle outline
		circle(result, center, radius, Scalar(0, 0, 255), 3, 8, 0);
	}
	return result;
}

Mat detectLines(Mat img) {
	vector<Vec4i> lines;
	Mat cannyImage, color_dest;
	Canny(img, cannyImage, 180, 100, 3);
	cvtColor(img, color_dest, CV_GRAY2BGR);
	HoughLinesP(cannyImage, lines, 1, CV_PI / 180, 25, 20, 3);

	for (size_t i = 0; i < lines.size(); i++) {
		line(color_dest, Point(lines[i][0], lines[i][1]), Point(lines[i][2], lines[i][3]), Scalar(150, 0, 0), 3, 8);
	}
	return color_dest;
}

vector<Vec4i> houghLinesFuncion(Mat img) {
	vector<Vec4i> lines;
	Mat cannyImage;
	Canny(img, cannyImage, 180, 100, 3);
	HoughLinesP(cannyImage, lines, 1, CV_PI / 180, 25, 20, 3);
	return lines;
}

Mat houghFunctionImage(Mat img) {
	Mat linesImage;
	cvtColor(img, linesImage, CV_GRAY2BGR);
	vector<Vec4i> lines = houghLinesFuncion(img);
	for (size_t i = 0; i < lines.size(); i++) {
		line(linesImage, Point(lines[i][0], lines[i][1]), Point(lines[i][2], lines[i][3]), Scalar(150, 0, 0), 3, 8);
	}
	return linesImage;
}

bool isInside(Mat img, int i, int j) {
	if ((i < img.rows && j < img.cols) && (i >= 0 && j >= 0)) {
		return true;
	}
	else {
		return false;
	}
}

Mat etichetare(Mat img) {
	Mat imgColor(img.rows, img.cols, CV_8UC3);

	label = 0;
	Mat labels = Mat::zeros(img.rows, img.cols, CV_32SC1);
	std::vector<std::vector<int>> edges;

	for (int i = 0; i < img.rows; i++)
	{
		for (int j = 0; j < img.cols; j++)
		{
			std::vector<int> L;
			if (img.at<uchar>(i, j) == 0 && labels.at<int>(i, j) == 0) {
				for (int k = 0; k < 4; k++) {
					if (isInside(img, i + di[k], j + dj[k])) {
						if (labels.at<int>(i + di[k], j + dj[k]) > 0) {
							L.push_back(labels.at<int>(i + di[k], j + dj[k]));
						}

					}

				}

				if (L.size() == 0) {
					label++;
					labels.at<int>(i, j) = label;
					edges.resize(label + 1);
				}
				else {
					int x = *min_element(L.begin(), L.end());
					labels.at<int>(i, j) = x;
					for (int y = 0; y < L.size(); y++)
					{
						if (L[y] != x) {
							edges[x].push_back(L[y]);
							edges[L[y]].push_back(x);
						}
					}
				}
			}
		}
	}
	std::default_random_engine gen;
	std::uniform_int_distribution<int> d(0, 140);
	for (int i = 1; i <= label; i++)
	{
		culoriRandom[i][0] = d(gen);
		culoriRandom[i][1] = d(gen);
		culoriRandom[i][2] = d(gen);
	}
	for (int i = 0; i < img.rows; i++)
	{
		for (int j = 0; j < img.cols; j++)
		{
			if (labels.at<int>(i, j) > 0) {
				imgColor.at<Vec3b>(i, j) = culoriRandom[labels.at<int>(i, j)];
			}
		}
	}

	int newLabels[1000] = {};
	newLabel = 0;

	for (int i = 0; i < label; i++) {
		if (newLabels[i] == 0) {
			newLabel++;
			std::queue<int> Q;
			newLabels[i] = newLabel;
			Q.push(i);
			while (!Q.empty()) {
				int x = Q.front();
				Q.pop();
				for (int y = 0; y < edges[x].size(); y++) {
					if (newLabels[edges[x][y]] == 0) {
						newLabels[edges[x][y]] = newLabel;
						Q.push(edges[x][y]);
					}
				}
			}
		}
	}
	for (int i = 0; i < img.rows; i++)
	{
		for (int j = 0; j < img.cols; j++)
		{
			if (labels.at<int>(i, j) > 0)
				labels.at<int>(i, j) = newLabels[labels.at<int>(i, j)];
		}
	}
	for (int i = 1; i <= label; i++)
	{
		culoriRandom[i][0] = d(gen);
		culoriRandom[i][1] = d(gen);
		culoriRandom[i][2] = d(gen);
	}
	for (int i = 0; i < img.rows; i++)
	{
		for (int j = 0; j < img.cols; j++)
		{
			if (labels.at<int>(i, j) > 0) {
				imgColor.at<Vec3b>(i, j) = culoriRandom[labels.at<int>(i, j)];
			}
		}
	}
	return imgColor;
}

vector<Mat> objectsFromImage(Mat img) {
	int height = img.rows;
	int width = img.cols;
	vector<Mat>imagesOfObjects;
	Mat imagineEtichetata = etichetare(img);
	//printf("Nr culori: %d\n", newLabel);
	for (int c = 0; c < newLabel; c++) {
		Mat object = Mat(height, width, CV_8UC3);
		for (int i = 0; i < height; i++) {
			for (int j = 0; j < width; j++) {
				if (imagineEtichetata.at<Vec3b>(i, j) == culoriRandom[c]) {
					object.at<Vec3b>(i, j) = imagineEtichetata.at<Vec3b>(i, j);
				}
			}
		}
		imagesOfObjects.push_back(object);
	}
	return imagesOfObjects;
}

vector<int> getNodes(vector<Vec3f> circles) {
	int V = circles.size();
	vector<int> *nodes = new vector<int>(V);
	return *nodes;
}

void setEdges(vector<Vec4i> edges) {
	for (int i = 0; i < edges.size(); i++) {
		//printf("Line P1=%d,%d P2=%d,%d\n", edges[i][0], edges[i][1], edges[i][2], edges[i][3]);
	}

}

bool filterDigit(Mat img) {
	int height = img.rows;
	int width = img.cols;
	Mat binarizeImage;

	int minX = img.cols, minY = img.rows, maxX = 0, maxY = 0;

	//convert image from RGB to GrayScale
	Mat src_gray = Mat(img.rows, img.cols, CV_8UC1);
	cvtColor(img, src_gray, CV_BGR2GRAY);

	binarizeImage = binaryImage(src_gray);

	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			if (binarizeImage.at<uchar>(i, j) == 0) {
				if (j < minX) minX = j;
				if (j > maxX) maxX = j;
				if (i < minY) minY = i;
				if (i > maxY) maxY = i;
			}
		}
	}

	printf("minX=%d, maxX=%d, minY=%d, maxY=%d\n", minX, maxX, minY, maxY);
	if (maxX - minX >= DIGIT_HIGHT_MIN && maxX - minX <= DIGIT_HIGHT_MAX) {
		if (maxY - minY >= DIGIT_WIDTH_MIN && maxY - minY <= DIGIT_WIDTH_MAX) {
			return true;
		}
	}
	return false;

}

vector<Mat> detectDigitImages(vector<Mat> images, int treashold) {
	vector<Mat> digits;
	Mat binarizeImage;
	int numberOfBlackPixels;
	for (int i = 0; i < images.size(); i++) {
		numberOfBlackPixels = 0;

		//convert image from RGB to GrayScale
		Mat src_gray = Mat(images[i].rows, images[i].cols, CV_8UC1);
		cvtColor(images[i], src_gray, CV_BGR2GRAY);

		binarizeImage = binaryImage(src_gray);
		for (int j = 0; j < binarizeImage.rows; j++) {
			for (int k = 0; k < binarizeImage.cols; k++) {
				if (binarizeImage.at<uchar>(j, k) == 0) {
					numberOfBlackPixels++;
				}
			}
		}
		if (numberOfBlackPixels >= treashold && filterDigit(images[i])) {
			digits.push_back(images[i]);
		}
	}
	return digits;
}

int main()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src, binarizedImg, edgeImg, circleImg, linesImage, imagineEtichetata;

		src = imread(fname, CV_LOAD_IMAGE_GRAYSCALE);
		//imshow("src", src);

		//binarise the initial image
		binarizedImg = binaryImage(src);
		//imshow("binarizedImg", binarizedImg);

		//detect edges
		Canny(binarizedImg, edgeImg, 180, 100, 3, false);
		//imshow("edgeImg", edgeImg);

		//detect circles
		circleImg = houghCirclesFuncionImage(src.clone());
		imshow("circleImg", circleImg);

		//detect lines
		linesImage = houghFunctionImage(binarizedImg.clone());
		imshow("Lines", linesImage);

		imagineEtichetata = etichetare(binarizedImg.clone());
		imshow("Imagine Etichetata", imagineEtichetata);

		//vector<Mat> objects = objectsFromImage(src);
		vector<Mat>::iterator it;
		vector<Vec4i> edges = houghLinesFuncion(binarizedImg.clone());
		setEdges(edges);
		auto objects = objectsFromImage(src.clone());
		/*for (int i = 0; i < objects.size(); i++) {
			imshow("Object", objects[i]);
			waitKey(500);
		}*/
		printf("Nr Objects %d\n", objects.size());
		for (int i = 0; i < newLabel; i++) {
			//printf("%d,%d,%d\n", culoriRandom[i][0], culoriRandom[i][1], culoriRandom[i][2]);
		}
		/*for (int i = 0; i < objects.size(); i++) {
			imshow("Object", objects[i]);
			waitKey(100);
		}*/
		auto digits = detectDigitImages(objects, 50);
		for (int i = 0; i < digits.size(); i++) {
			imshow("Object", digits[i]);
			waitKey(200);
		}



		waitKey();
	}
	return 0;
}