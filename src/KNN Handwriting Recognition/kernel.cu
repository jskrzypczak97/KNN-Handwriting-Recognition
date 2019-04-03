#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/core.hpp"
#include <iostream>
#include <string>
#include <vector>
#include <stdio.h>
#include <list>
#include <fstream>
#include <algorithm>
#include <chrono>
#include <iomanip>

#ifndef __CUDACC__ 
#define __CUDACC__
#endif
#include <device_functions.h>

using namespace std;

// DEFINING CONSTANTS
const int LINE_WS_CONST = 2; // White sign adjustment used for files reading
const int ASCII_LABEL_CONST = 97; // ASCII to integer adjustment

const int NUM_OF_CLASSES = 26; // Number of letters
const int IMAGE_SIZE = 128; // In pixels
const int IMAGE_VEC_SIZE = IMAGE_SIZE * IMAGE_SIZE; // Because images are square

const int K_NN = 100; // Number of nearest neighbours
const int N = 64; // Number of images that are read in single GPU usage

const int THREADS_NUM = 1024;
const int BLOCKS_NUM_1 = IMAGE_VEC_SIZE * N / THREADS_NUM;
const int BLOCKS_NUM_2 = BLOCKS_NUM_1 * N / THREADS_NUM;

struct Match
{
	double match_per;
	int label;
};

struct Result
{
	int num_of_class_members;
	int label;
};

bool key_sort_Match(const Match& a, const Match& b)
{
	return a.match_per > b.match_per;
}

bool key_sort_Result(const Result& a, const Result& b)
{
	return a.num_of_class_members > b.num_of_class_members;
}

__global__ void ChceckMatchKernel(char * input_img, char * dataset_imgs, int * image_matches)
{
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	int offset = index / IMAGE_VEC_SIZE;
	if (index < IMAGE_VEC_SIZE*N) 
		(input_img[index - IMAGE_VEC_SIZE * offset] == 48 && input_img[index - IMAGE_VEC_SIZE * offset] == dataset_imgs[index]) ?
		image_matches[index] = 1 : image_matches[index] = 0;
}

__global__ void SumKernel(int * input_matches, int * output_matches)
{
	__shared__ int shared_mem[THREADS_NUM];
	
	int thread_id = threadIdx.x;
	int index = blockDim.x * blockIdx.x + threadIdx.x;
	shared_mem[thread_id] = input_matches[index];

	__syncthreads();

	for (int step = blockDim.x / 2; step > 0; step >>= 1)
	{
		if (thread_id < step) shared_mem[thread_id] += shared_mem[thread_id + step];
		__syncthreads();
	}
	if (thread_id == 0)
		output_matches[blockIdx.x] = shared_mem[0];
}

int getFileSize(string);
void getInputImage(string,char[]);
void ClearVector(char*,int);
void AssignNeighbours(int[],list<Match>&,int,int);
void getResults(list<Match>&, list<Result>&);
void printResults(double, list<Result>&);
double CPU_recognition(char *, char *, list<Match>&);
double GPU_recognition(char *, char *, list<Match>&);

int main()
{
	string image_name;
	string cpu_gpu;

	list<Match> Matches;
	list<Result> Top_classes;

	cout << "Type C for CPU or G for GPU: ";
	cin >> cpu_gpu;
	cout << endl;
	cout << "Image name: ";
	cin >> image_name;
	system("CLS");
	cout << "Loading..." << endl;

	char * input_img = new char[IMAGE_VEC_SIZE]();
	getInputImage(image_name, input_img);

	char * dataset_imgs = new char[IMAGE_VEC_SIZE*N]();

	system("CLS");
	cout << "Loading..." << endl;

	double full_time = 0;

	if (cpu_gpu == "C")
	{
		full_time = CPU_recognition(input_img, dataset_imgs, Matches);
	}
	else if(cpu_gpu == "G")
	{
		full_time = GPU_recognition(input_img, dataset_imgs, Matches);
	}
	else
	{
		cout << "Nope" << endl;
		return -1;
	}

	getResults(Matches, Top_classes);
	printResults(full_time, Top_classes);

	return 0;
}

int getFileSize(string fileName)
{
	ifstream file(fileName.c_str(), ifstream::in | ifstream::binary);

	if (!file.is_open())
	{
		return -1;
	}

	file.seekg(0, ios::end);
	int fileSize = file.tellg();
	file.close();

	return fileSize;
}

void ClearVector(char * vec, int size)
{
	for (int i = 0; i < size; i++) vec[i] = 0;
}

void getInputImage(string image_name, char input_img[])
{
	// READING INPUT IMAGE
	cv::Mat img = cv::imread(image_name, 0);			// 0 for grayscale
	cv::Mat img_bin;
	cv::threshold(img, img_bin, 0, 1, 0);			// binarising image

	std::vector<uchar> vector_img;
	if (img_bin.isContinuous()) vector_img.assign(img_bin.datastart, img_bin.dataend);	 // assigning binarised image to vector as we want it in 1D
	//copy(vector_img.begin(), vector_img.end(), input_img);
																						 
	int input_img_ocv[IMAGE_VEC_SIZE];								// finall form of input image that we will compare to dataset 
	copy(vector_img.begin(), vector_img.end(), input_img_ocv);

	ofstream o_input_file;
	o_input_file.open("inputimg.txt");

	for (int i = 0; i < IMAGE_VEC_SIZE; i++) o_input_file << input_img_ocv[i];

	o_input_file.close();

	ifstream i_input_file;
	i_input_file.open("inputimg.txt");

	for (int i = 0; i < IMAGE_VEC_SIZE; i++) i_input_file >> input_img[i];

	i_input_file.close();
}

void AssignNeighbours(int matches_array[],list<Match>& Matches, int label, int I)
{
	for (int i = 0; i < I; i++)
	{
		Match neighbour;
		neighbour.label = label;
		neighbour.match_per = (double)matches_array[i] * 100 / (double)IMAGE_VEC_SIZE;
		Matches.push_front(neighbour);
	}
}

void getResults(list<Match>& Matches, list<Result>& Top_classes)
{
	Matches.sort(key_sort_Match);

	int * k_nn = new int[NUM_OF_CLASSES]();
	int index = Matches.front().label - ASCII_LABEL_CONST;

	for (int i = 0; i < K_NN; i++)
	{
		index = Matches.front().label - ASCII_LABEL_CONST;
		k_nn[index]++;
		Matches.pop_front();
	}

	for (int i = 0; i < NUM_OF_CLASSES; i++)
	{
		if (k_nn[i] != 0)
		{
			Result result;
			result.num_of_class_members = k_nn[i];
			result.label = i + ASCII_LABEL_CONST;
			Top_classes.push_front(result);
		}
	}

	Top_classes.sort(key_sort_Result);
}

void printResults(double full_time, list<Result>& Top_classes)
{
	cout << endl << "Recognised letter is '" << (char)Top_classes.front().label << "'" << endl << endl;

	cout << "TOP 3:" << endl;
	cout << "1. Letter: " << (char)Top_classes.front().label << "	" << (float)Top_classes.front().num_of_class_members / (float)K_NN * 100 << "%" << endl;
	Top_classes.pop_front();
	cout << "2. Letter: " << (char)Top_classes.front().label << "	" << (float)Top_classes.front().num_of_class_members / (float)K_NN * 100 << "%" << endl;
	Top_classes.pop_front();
	cout << "3. Letter: " << (char)Top_classes.front().label << "	" << (float)Top_classes.front().num_of_class_members / (float)K_NN * 100 << "%" << endl << endl;

	cout << "It took " << setprecision(3) << full_time << " s" << endl;
	system("PAUSE");
}

double CPU_recognition(char * input_img, char * dataset_imgs, list<Match>& Matches)
{
	string file_name = "+_elodataset.txt";
	string full_path = "";
	string line;
	int dataset_file_length;
	int dataset_images_size;
	int dataset_label = 0;
	ifstream dataset_file;

	chrono::high_resolution_clock::time_point single_file_step_start;
	chrono::high_resolution_clock::time_point single_file_step_stop;
	chrono::duration<double> single_file_step_time = chrono::duration_cast<chrono::duration<double>>(single_file_step_stop - single_file_step_start);
	double step_sum_time = 0;
	double time = 0;

	int match[1] = { 0 }; // Single value array to make it compatible with AssignNeighbours()

	for (int i = 0; i < NUM_OF_CLASSES; i++)
	{
		file_name[0] = ASCII_LABEL_CONST + i;							// ASCII trick to get proper name of the file in each loop
		full_path = "DATASET_ELO\\" + file_name;

		dataset_file_length = getFileSize(full_path);

		dataset_file.open(full_path.c_str());

		if (!dataset_file.is_open())
		{
			cout << "Could not open dataset file" << endl;
			return 0;
		}

		getline(dataset_file, line);
		dataset_label = stoi(line);

		cout << "Analizing letter '" << (char)dataset_label << "'...";

		while (!dataset_file.eof())
		{
			for (int i = 0; i < IMAGE_VEC_SIZE; i++) dataset_file >> dataset_imgs[i];
			if (dataset_file.eof()) break;

			single_file_step_start = chrono::high_resolution_clock::now();

			for (int i = 0; i < IMAGE_VEC_SIZE; i++)
			{
				if (input_img[i] == dataset_imgs[i] && input_img[i] == 48)
				{
					match[0]++;
				}
			}

			single_file_step_stop = chrono::high_resolution_clock::now();

			single_file_step_time = chrono::duration_cast<chrono::duration<double>>(single_file_step_stop - single_file_step_start);
			step_sum_time += single_file_step_time.count();

			AssignNeighbours(match, Matches, dataset_label, 1);
			match[0] = 0;
		}

		cout << fixed;
		cout << " done in " << setprecision(3) << step_sum_time * 1000 << " ms" << endl;
		time += step_sum_time;
		step_sum_time = 0;

		dataset_file.close();
	}
	return time;
}

double GPU_recognition(char * input_img, char * dataset_imgs, list<Match>& Matches)
{
	string file_name = "+_elodataset.txt";
	string full_path = "";
	string line;
	int dataset_file_length;
	int dataset_images_size;
	int dataset_label = 0;
	ifstream dataset_file;

	chrono::high_resolution_clock::time_point single_file_step_start;
	chrono::high_resolution_clock::time_point single_file_step_stop;
	chrono::duration<double> single_file_step_time = chrono::duration_cast<chrono::duration<double>>(single_file_step_stop - single_file_step_start);
	double time = 0;
	double step_sum_time = 0;

	int * image_matches = new int[IMAGE_VEC_SIZE*N]();
	int * matches_first_sum = new int[BLOCKS_NUM_1]();
	int * matches_final_sum = new int[BLOCKS_NUM_2]();
	int adjust_loop = 0;
	int num_of_matches;

	char * dev_dataset_images;
	char * dev_input_image;
	int * dev_image_matches;
	int * dev_matches_first_sum;
	int * dev_matches_final_sum;

	cudaMalloc(&dev_dataset_images, sizeof(char)*IMAGE_VEC_SIZE*N);
	cudaMalloc(&dev_input_image, sizeof(char)*IMAGE_VEC_SIZE);
	cudaMalloc(&dev_image_matches, sizeof(int)*IMAGE_VEC_SIZE*N);
	cudaMalloc(&dev_matches_first_sum, sizeof(int)*BLOCKS_NUM_1);
	cudaMalloc(&dev_matches_final_sum, sizeof(int)*BLOCKS_NUM_2);

	cudaMemcpy(dev_input_image, input_img, sizeof(char)*IMAGE_VEC_SIZE, cudaMemcpyHostToDevice);
	cudaMemcpy(dev_image_matches, image_matches, sizeof(int)*IMAGE_VEC_SIZE*N, cudaMemcpyHostToDevice);
	cudaMemcpy(dev_matches_first_sum, matches_first_sum, sizeof(int)*BLOCKS_NUM_1, cudaMemcpyHostToDevice);
	cudaMemcpy(dev_matches_final_sum, matches_final_sum, sizeof(int)*BLOCKS_NUM_2, cudaMemcpyHostToDevice);

	for (int i = 0; i < NUM_OF_CLASSES; i++)
	{
		file_name[0] = ASCII_LABEL_CONST + i;							// ASCII trick to get proper name of the file in each loop
		full_path = "DATASET_ELO\\" + file_name;

		dataset_file_length = getFileSize(full_path);

		dataset_file.open(full_path.c_str());

		if (!dataset_file.is_open())
		{
			cout << "Could not open dataset file" << endl;
			return 0;
		}

		getline(dataset_file, line);
		dataset_label = stoi(line);

		cout << "Analizing letter '" << (char)dataset_label << "'...";

		dataset_images_size = dataset_file_length - line.length();
		adjust_loop = 0;

		for (int j = 0; j < dataset_images_size; j += (IMAGE_VEC_SIZE + LINE_WS_CONST) * N)
		{
			if (j + (IMAGE_VEC_SIZE + LINE_WS_CONST) * N > dataset_images_size)
			{
				adjust_loop = IMAGE_VEC_SIZE * N - dataset_images_size + j;
				ClearVector(dataset_imgs, IMAGE_VEC_SIZE * N);
			}

			for (int k = 0; k < IMAGE_VEC_SIZE * N - adjust_loop; k++) dataset_file >> dataset_imgs[k];

			cudaMemcpy(dev_dataset_images, dataset_imgs, sizeof(char)*(IMAGE_VEC_SIZE*N), cudaMemcpyHostToDevice);

			single_file_step_start = chrono::high_resolution_clock::now();

			ChceckMatchKernel << < BLOCKS_NUM_1, THREADS_NUM >> > (dev_input_image, dev_dataset_images, dev_image_matches);
			SumKernel << < BLOCKS_NUM_1, THREADS_NUM >> > (dev_image_matches, dev_matches_first_sum);
			SumKernel << <BLOCKS_NUM_2, THREADS_NUM / N >> > (dev_matches_first_sum, dev_matches_final_sum);

			single_file_step_stop = chrono::high_resolution_clock::now();
			single_file_step_time = chrono::duration_cast<chrono::duration<double>>(single_file_step_stop - single_file_step_start);

			cudaMemcpy(matches_final_sum, dev_matches_final_sum, sizeof(int)*(BLOCKS_NUM_2), cudaMemcpyDeviceToHost);

			step_sum_time += single_file_step_time.count();

			num_of_matches = floor((((float)IMAGE_VEC_SIZE * N - (float)adjust_loop) / (float)IMAGE_VEC_SIZE));

			AssignNeighbours(matches_final_sum, Matches, dataset_label, num_of_matches);
		}
		cout << fixed;
		cout << " done in " << setprecision(3) << step_sum_time * 1000 << " ms" << endl;

		dataset_file.close();
		time += step_sum_time;
		step_sum_time = 0;
	}

	cudaFree(dev_image_matches);
	cudaFree(dev_input_image);
	cudaFree(dev_dataset_images);
	cudaFree(dev_matches_first_sum);
	cudaFree(dev_matches_final_sum);

	delete[] dataset_imgs, image_matches, matches_first_sum, matches_final_sum;

	return time;
}