#include "stdafx.h"
#include "common.h"
#include <vector>
#include <iostream>
#include<sstream>
#include <stdio.h>
#include<random>
#include <algorithm>
#include <opencv/cv.h>
#include<opencv2/ml/ml.hpp>
#include <opencv/highgui.h>
#include<opencv2/core/core.hpp>
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <list>

using namespace std;
using namespace cv;
using namespace std;

#define DIGIT_HIGHT_MIN 15
#define DIGIT_HIGHT_MAX 50

#define DIGIT_WIDTH_MIN 15
#define DIGIT_WIDTH_MAX 50

#define	LINE_CIRCLE_DISTANCE 75

#define MIN_CONTOUR_AREA 100

#define RESIZED_IMAGE_WIDTH 20
#define RESIZED_IMAGE_HEIGHT 30

#define DISTANCE_BTW_CENTERS 30

typedef struct nodeInfo {
	Point position;
	int value;
}NodeInfo;

typedef struct node {
	Point circle;
	NodeInfo info;
}Node_;

int di[4] = { 0, -1, -1, -1 };
int dj[4] = { -1, -1, 0, +1, };

Vec3b culoriRandom[1000];
int label;
int newLabel;

vector<NodeInfo> graphNodes;
NodeInfo currentNode;
vector<Node_> nodes;
vector<Point> centerCircles;

int **adjacencyMatrix;

class ContourWithData {
public:
	std::vector<cv::Point> ptContour;           // contour
	cv::Rect boundingRect;                      // bounding rect for contour
	float fltArea;                              // area of contour

	bool checkIfContourIsValid() {                              // obviously in a production grade program
		if (fltArea < MIN_CONTOUR_AREA) return false;           // we would have a much more robust function for 
		return true;                                            // identifying if a contour is valid !!
	}

	static bool sortByBoundingRectXPosition(const ContourWithData& cwdLeft, const ContourWithData& cwdRight) {      // this function allows us to sort
		return(cwdLeft.boundingRect.x < cwdRight.boundingRect.x);                                                   // the contours from left to right
	}
};

void addEdge(vector<int> adj[], int u, int v)
{
	adj[u].push_back(v);
	adj[v].push_back(u);
}

// A utility function to print the adjacency list 
// representation of graph 
void printGraph(vector<int> adj[], int V)
{
	for (int v = 0; v < V; ++v)
	{
		cout << "\n " << v << " : ";
		for (auto x : adj[v])
			cout << "-> " << x;
	}
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
	HoughLinesP(cannyImage, lines, 1, CV_PI / 180, 35, 80, 6);
	return lines;
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
		centerCircles.push_back(center);
		int radius = cvRound(circles[i][2]);
		// circle center
		circle(circlesImage, center, 3, Scalar(0, 255, 255), -1, 8, 0);
		// circle outline
		circle(circlesImage, center, radius, Scalar(250, 0, 255), 3, 8, 0);
	}
	return circlesImage;
}

Point computeCenter(Vec4i line) {
	//printf("line xmin %d xmax %d ymin%d ymax%d centerX%d centeY%d\n", line[0], line[2], line[3], line[1], line[0] + abs(line[2] - line[0]) / 2, line[3] + abs(line[3] - line[1]) / 2);
	return Point(line[0] + abs(line[2] - line[0]) / 2, line[3] + abs(line[3] - line[1]) / 2);
}

vector<Vec4i> detectRealLines(vector<Vec4i> lines) {
	vector<Vec4i> result;
	for (int i = 0; i < lines.size(); i++) {
		for (int j = 0; j < lines.size(); j++) {
			if ((i != j) && (abs(computeCenter(lines[i]).x - computeCenter(lines[j]).x) < 5 || abs(computeCenter(lines[i]).y - computeCenter(lines[j]).y) < 5)) {
				lines.erase(lines.begin() + j);
			}

		}
		result.push_back(lines[i]);
	}
	return result;
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
	std::uniform_int_distribution<int> d(100, 140);
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

Point computePosition(int maxX, int minX, int maxY, int minY) {
	return Point(minX + abs(maxX - minX) / 2, minY + abs(maxY - minY) / 2);
}

bool isDigit(Mat img) {
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

	//printf("minX=%d, maxX=%d, minY=%d, maxY=%d\n", minX, maxX, minY, maxY);
	if (maxX - minX >= DIGIT_HIGHT_MIN && maxX - minX <= DIGIT_HIGHT_MAX) {
		if (maxY - minY >= DIGIT_WIDTH_MIN && maxY - minY <= DIGIT_WIDTH_MAX) {
			currentNode.position = computePosition(maxX, minX, maxY, minY);
			return true;
		}
	}
	return false;
}

void bindCircleInfo() {
	for (int i = 0; i < centerCircles.size(); i++) {
		for (int j = 0; j < graphNodes.size(); j++) {
			//euclidian distance between 2 centers
			if ((abs(centerCircles[i].x - graphNodes[j].position.x) + abs(centerCircles[i].y - graphNodes[j].position.y)) < DISTANCE_BTW_CENTERS) {
				Node_ node_;
				node_.circle = centerCircles[i];
				//TO DO remove found value from graphNodes
				node_.info = graphNodes[j];
				nodes.push_back(node_);
			}
		}
	}
}

int characterRecognision(Mat characterImage) {
	int value = -1;
	std::vector<ContourWithData> allContoursWithData;           // declare empty vectors,
	std::vector<ContourWithData> validContoursWithData;         // we will fill these shortly

																// read in training classifications ///////////////////////////////////////////////////

	cv::Mat matClassificationInts;      // we will read the classification numbers into this variable as though it is a vector

	cv::FileStorage fsClassifications("classifications.xml", cv::FileStorage::READ);        // open the classifications file

	if (fsClassifications.isOpened() == false) {                                                    // if the file was not opened successfully
		std::cout << "error, unable to open training classifications file, exiting program\n\n";    // show error message
	}

	fsClassifications["classifications"] >> matClassificationInts;      // read classifications section into Mat classifications variable
	fsClassifications.release();                                        // close the classifications file

																		// read in training images ////////////////////////////////////////////////////////////

	cv::Mat matTrainingImagesAsFlattenedFloats;         // we will read multiple images into this single image variable as though it is a vector

	cv::FileStorage fsTrainingImages("images.xml", cv::FileStorage::READ);          // open the training images file

	if (fsTrainingImages.isOpened() == false) {                                                 // if the file was not opened successfully
		std::cout << "error, unable to open training images file, exiting program\n\n";         // show error message
	}

	fsTrainingImages["images"] >> matTrainingImagesAsFlattenedFloats;           // read images section into Mat training images variable
	fsTrainingImages.release();                                                 // close the traning images file

																				// train //////////////////////////////////////////////////////////////////////////////

	cv::Ptr<cv::ml::KNearest>  kNearest(cv::ml::KNearest::create());            // instantiate the KNN object

																				// finally we get to the call to train, note that both parameters have to be of type Mat (a single Mat)
																				// even though in reality they are multiple images / numbers
	kNearest->train(matTrainingImagesAsFlattenedFloats, cv::ml::ROW_SAMPLE, matClassificationInts);

	// test ///////////////////////////////////////////////////////////////////////////////

	cv::Mat matTestingNumbers = characterImage;            // read in the test numbers image

	if (matTestingNumbers.empty()) {                                // if unable to open image
		std::cout << "error: image not read from file\n\n";         // show error message on command line
	}

	cv::Mat matGrayscale;           //
	cv::Mat matBlurred;             // declare more image variables
	cv::Mat matThresh;              //
	cv::Mat matThreshCopy;          //

	cv::cvtColor(matTestingNumbers, matGrayscale, CV_BGR2GRAY);         // convert to grayscale

																		// blur
	cv::GaussianBlur(matGrayscale,              // input image
		matBlurred,                // output image
		cv::Size(5, 5),            // smoothing window width and height in pixels
		0);                        // sigma value, determines how much the image will be blurred, zero makes function choose the sigma value

								   // filter image from grayscale to black and white
	cv::adaptiveThreshold(matBlurred,                           // input image
		matThresh,                            // output image
		255,                                  // make pixels that pass the threshold full white
		cv::ADAPTIVE_THRESH_GAUSSIAN_C,       // use gaussian rather than mean, seems to give better results
		cv::THRESH_BINARY_INV,                // invert so foreground will be white, background will be black
		11,                                   // size of a pixel neighborhood used to calculate threshold value
		2);                                   // constant subtracted from the mean or weighted mean

	matThreshCopy = matThresh.clone();              // make a copy of the thresh image, this in necessary b/c findContours modifies the image

	std::vector<std::vector<cv::Point> > ptContours;        // declare a vector for the contours
	std::vector<cv::Vec4i> v4iHierarchy;                    // declare a vector for the hierarchy (we won't use this in this program but this may be helpful for reference)

	cv::findContours(matThreshCopy,             // input image, make sure to use a copy since the function will modify this image in the course of finding contours
		ptContours,                             // output contours
		v4iHierarchy,                           // output hierarchy
		cv::RETR_EXTERNAL,                      // retrieve the outermost contours only
		cv::CHAIN_APPROX_SIMPLE);               // compress horizontal, vertical, and diagonal segments and leave only their end points

	for (int i = 0; i < ptContours.size(); i++) {               // for each contour
		ContourWithData contourWithData;                                                    // instantiate a contour with data object
		contourWithData.ptContour = ptContours[i];                                          // assign contour to contour with data
		contourWithData.boundingRect = cv::boundingRect(contourWithData.ptContour);         // get the bounding rect
		contourWithData.fltArea = cv::contourArea(contourWithData.ptContour);               // calculate the contour area
		allContoursWithData.push_back(contourWithData);                                     // add contour with data object to list of all contours with data
	}

	for (int i = 0; i < allContoursWithData.size(); i++) {                      // for all contours
		if (allContoursWithData[i].checkIfContourIsValid()) {                   // check if valid
			validContoursWithData.push_back(allContoursWithData[i]);            // if so, append to valid contour list
		}
	}
	// sort contours from left to right
	std::sort(validContoursWithData.begin(), validContoursWithData.end(), ContourWithData::sortByBoundingRectXPosition);

	std::string strFinalString;         // declare final string, this will have the final number sequence by the end of the program

	for (int i = 0; i < validContoursWithData.size(); i++) {            // for each contour

																		// draw a green rect around the current char
		cv::rectangle(matTestingNumbers,                            // draw rectangle on original image
			validContoursWithData[i].boundingRect,        // rect to draw
			cv::Scalar(0, 255, 0),                        // green
			2);                                           // thickness

		cv::Mat matROI = matThresh(validContoursWithData[i].boundingRect);          // get ROI image of bounding rect

		cv::Mat matROIResized;
		cv::resize(matROI, matROIResized, cv::Size(RESIZED_IMAGE_WIDTH, RESIZED_IMAGE_HEIGHT));     // resize image, this will be more consistent for recognition and storage

		cv::Mat matROIFloat;
		matROIResized.convertTo(matROIFloat, CV_32FC1);             // convert Mat to float, necessary for call to find_nearest

		cv::Mat matROIFlattenedFloat = matROIFloat.reshape(1, 1);

		cv::Mat matCurrentChar(0, 0, CV_32F);

		kNearest->findNearest(matROIFlattenedFloat, 1, matCurrentChar);     // finally we can call find_nearest !!!
		//TO DO

		float fltCurrentChar = (float)matCurrentChar.at<float>(0, 0);

		value = int(fltCurrentChar) - 48;
		strFinalString = strFinalString + char(int(fltCurrentChar));        // append current char to full string
	}

	std::cout << "numbers read = " << strFinalString << "\n";       // show the full string

	cv::imshow("matTestingNumbers", matTestingNumbers);     // show input image with green boxes drawn around found digits

	cv::waitKey(10);
	return value;
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

		if (numberOfBlackPixels >= treashold && isDigit(images[i])) {
			currentNode.value = characterRecognision(images[i]);
			graphNodes.push_back(currentNode);
			digits.push_back(images[i]);
		}
	}
	return digits;
}

void initializeAdiacentMatrix(int size) {
	adjacencyMatrix = new int*[size];
	for (int i = 0; i < size; ++i)
		adjacencyMatrix[i] = new int[size];

	for (int i = 0; i < 10; i++) {
		for (int j = 0; j < 10; j++) {
			adjacencyMatrix[i][j] = 0;
		}
	}
}

void bindLineWithCircles(Vec4i line) {
	for (int i = 0; i < nodes.size() - 1; i++) {
		if ((((abs(nodes[i].circle.x - line.val[0]) < LINE_CIRCLE_DISTANCE) && (abs(nodes[i].circle.y - line.val[1]) < LINE_CIRCLE_DISTANCE))) || ((abs(nodes[i].circle.x - line.val[2]) < LINE_CIRCLE_DISTANCE) && (abs(nodes[i].circle.y - line.val[3]) < LINE_CIRCLE_DISTANCE))) {
			for (int j = i + 1; j < nodes.size(); j++) {
				if ((((abs(nodes[j].circle.x - line.val[0]) < LINE_CIRCLE_DISTANCE) && (abs(nodes[j].circle.y - line.val[1]) < LINE_CIRCLE_DISTANCE))) || ((abs(nodes[j].circle.x - line.val[2]) < LINE_CIRCLE_DISTANCE) && (abs(nodes[j].circle.y - line.val[3]) < LINE_CIRCLE_DISTANCE))) {


					adjacencyMatrix[nodes[i].info.value][nodes[j].info.value] = 1;
					adjacencyMatrix[nodes[j].info.value][nodes[i].info.value] = 1;

				}
			}
		}
	}
}

void digitalizeGraph(Mat img) {
	initializeAdiacentMatrix(10);
	vector<Vec4i> lines = houghLinesFuncion(img);
	vector<Vec4i> realLines = detectRealLines(lines);
	Mat linesImage;
	cvtColor(img, linesImage, CV_GRAY2BGR);
	for (int i = 0; i < realLines.size(); i++) {
		line(linesImage, Point(realLines[i][0], realLines[i][1]), Point(realLines[i][2], realLines[i][3]), Scalar(0, 0, 255), 3, 8);
		imshow("Real Lines", linesImage);
		waitKey(20);
	}
	for (int i = 0; i < realLines.size(); i++) {
		bindLineWithCircles(realLines[i]);
	}

}

void printAdjacencyMatrix() {
	printf("\nAdjacency Matrix:\n");
	for (int i = 0; i < 10; i++) {
		for (int j = 0; j < 10; j++) {
			printf("%d ", adjacencyMatrix[i][j]);
		}
		printf("\n");
	}
	printf("\n");
}

void convertAdjacencyMatrixToAdjacencyList() {
	vector<int> adj[10];
	for (int i = 0; i < 10; i++) {
		for (int j = 0; j < 10; j++) {
			if ((adjacencyMatrix[i][j] == 1) && (i < j)) {
				addEdge(adj, i, j);
			}
		}
	}
	printf("\nAdjacency Matrix:");
	printGraph(adj, 10);
}
int main() {

	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src, binarizedImg, edgeImg, circleImg, linesImage, imagineEtichetata;

		src = imread(fname, CV_LOAD_IMAGE_GRAYSCALE);

		binarizedImg = binaryImage(src);

		circleImg = houghCirclesFuncionImage(src.clone());
		imshow("Real Circles", circleImg);

		imagineEtichetata = etichetare(binarizedImg.clone());
		imshow("Labeled Image", imagineEtichetata);

		vector<Mat>::iterator it;
		vector<Vec4i> edges = houghLinesFuncion(binarizedImg.clone());
		setEdges(edges);
		auto objects = objectsFromImage(src.clone());
		printf("Nr Objects: %d\n", objects.size());

		auto digits = detectDigitImages(objects, 50);

		bindCircleInfo();

		for (int i = 0; i < nodes.size(); i++) {
			printf("CircleX=%d, CircleY=%d, Value=%d\n", nodes[i].circle.x, nodes[i].circle.y, nodes[i].info.value);
		}

		digitalizeGraph(binarizedImg.clone());

		printAdjacencyMatrix();
	
		convertAdjacencyMatrixToAdjacencyList();

		waitKey();
	}

	return(0);
}


