#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <filesystem>

using namespace cv;
using namespace std;
namespace fs = std::filesystem;




// Purpose: Computes the energy map of the input image using Sobel operators to highlight edges.
// Parameters:
//   - image: Input color image (Mat) in BGR format
// Returns:
//   - energy: Energy map (Mat) with gradient magnitudes

Mat computeEnergy(const Mat& image) {
    Mat gray, grad_x, grad_y, abs_grad_x, abs_grad_y, energy;
    cvtColor(image, gray, COLOR_BGR2GRAY);
    Sobel(gray, grad_x, CV_64F, 1, 0, 3);
    Sobel(gray, grad_y, CV_64F, 0, 1, 3);
    abs_grad_x = abs(grad_x);
    abs_grad_y = abs(grad_y);
    energy = abs_grad_x + abs_grad_y;
    return energy;
}


// Purpose: Identifies the lowest-energy vertical seam using dynamic programming.
// Parameters:
//   - energy: Energy map (Mat) of the image
// Returns:
//   - seam: Vector of column indices representing the seam path

vector<int> findSeam(const Mat& energy) {
    int rows = energy.rows, cols = energy.cols;
    Mat cost = energy.clone();
    Mat backtrack(rows, cols, CV_32S);
    for (int i = 1; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            double minCost = cost.at<double>(i - 1, j);
            int offset = 0;
            if (j > 0 && cost.at<double>(i - 1, j - 1) < minCost) { minCost = cost.at<double>(i - 1, j - 1); offset = -1; }
            if (j < cols - 1 && cost.at<double>(i - 1, j + 1) < minCost) { minCost = cost.at<double>(i - 1, j + 1); offset = 1; }
            cost.at<double>(i, j) += minCost;
            backtrack.at<int>(i, j) = offset;
        }
    }
    vector<int> seam(rows);
    seam[rows - 1] = min_element(cost.ptr<double>(rows - 1), cost.ptr<double>(rows - 1) + cols) - cost.ptr<double>(rows - 1);
    for (int i = rows - 2; i >= 0; i--) {
        seam[i] = seam[i + 1] + backtrack.at<int>(i + 1, seam[i + 1]);
    }
    return seam;
}




// Purpose: Removes a specified vertical seam from the image, reducing its width by 1 pixel.
// Parameters:
//   - image: Input image (Mat) in BGR format
//   - seam: Vector of column indices to remove (one per row)
// Returns:
//   - output: Image with the seam removed (width reduced by 1)

Mat removeSeam(const Mat& image, const vector<int>& seam) {
    int rows = image.rows, cols = image.cols;
    Mat output(rows, cols - 1, CV_8UC3);
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < seam[i]; j++) {
            output.at<Vec3b>(i, j) = image.at<Vec3b>(i, j);
        }
        for (int j = seam[i] + 1; j < cols; j++) {
            output.at<Vec3b>(i, j - 1) = image.at<Vec3b>(i, j);
        }
    }
    return output;
}


// Purpose: Reduces the image width by removing multiple low-energy vertical seams.
// Parameters:
//   - image: Input image (Mat) to be resized
//   - numSeams: Number of seams to remove (i.e., pixels to reduce from width)
// Returns:
//   - image: Resized image after seam removal
Mat seamCarving(Mat image, int numSeams) {
    if (numSeams >= image.cols) {
        cout << "Error: numSeams (" << numSeams << ") exceeds image width (" << image.cols << ")!" << endl;
        return image.clone();
    }
    for (int i = 0; i < numSeams; i++) {
        Mat energy = computeEnergy(image);
        vector<int> seam = findSeam(energy);
        image = removeSeam(image, seam);
    }
    return image;
}

int main() {
    string inputImage;
    cout << "Enter the input image filename (e.g., image.png or full path): ";
    getline(cin, inputImage);

    Mat image = imread(inputImage);

    if (image.empty()) {
        cout << "Error: Could not open image '" << inputImage << "'!" << endl;
        return -1;
    }

    cout << "Original image size: " << image.cols << "x" << image.rows << endl;

    int numSeams;
    cout << "Enter the number of seams to remove: ";
    cin >> numSeams;

    Mat resizedImage = seamCarving(image, numSeams);
    if (resizedImage.empty()) {
        cout << "Error: resizedImage is empty!" << endl;
        return -1;
    }
    cout << "Resized image size: " << resizedImage.cols << "x" << resizedImage.rows << endl;

    string outputImage;
    cout << "Enter the output image filename (e.g., resized.png or full path, press Enter for 'resized.png'): ";
    cin.ignore();  // Clear the newline from the previous input
    getline(cin, outputImage);
    if (outputImage.empty()) {
        outputImage = "resized.png";
    }

    // Determine if the output path is a directory or file
    string outputPath = outputImage;
    if (outputImage.find_last_of("/\\") != string::npos) {
        outputPath = outputImage.substr(0, outputImage.find_last_of("/\\") + 1);
    }
    else {
        outputPath = "./";  // Default to current directory
    }
    fs::create_directories(outputPath);

    if (!imwrite(outputImage, resizedImage)) {
        cout << "Error: Failed to save " << outputImage << "!" << endl;
        return -1;
    }
    cout << "Seam carving completed. Saved as " << outputImage << endl;

    return 0;
}