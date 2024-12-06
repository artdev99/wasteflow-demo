#ifndef MASK_TOOLS_H
#define MASK_TOOLS_H

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <string>
#include <vector>

//remove magic numbers
#define BLUE_POS          0
#define GREEN_POS         1
#define RED_POS           2
#define ALPHA_POS         3
#define WHITE_BACKGROUND  255
#define SQUARE_SIZE       300

using namespace std;
using namespace cv;

//bgr format
struct pixel{
    uchar b;
    uchar g;
    uchar r;
};

//predefined colors (22 details)
const pixel blue = {255,0,0}; //mirror, 0000ff
const pixel green = {0,255,0}; //rail, 00ff00
const pixel red = {0,0,255}; //wheel, ff0000
const pixel yellow = {0,255,255}; //front door, ffff00
const pixel cyan = {255,255,0}; //backdoor, 00ffff
const pixel magenta = {255,0,255}; //window, ff00ff
const pixel orange = {00,128,255}; //lights, ff8000
const pixel purple = {255,122,131}; //license plate, 837aff
const pixel light_green = {156,224,141}; //front side, 8de09c
const pixel brown = {93,107,169}; //back side, a96b5d
const pixel pastel_red = {89,84,240}; //car front, f05459
const pixel light_purple = {255,154,233}; //car back, e99aff
const pixel bright_green = {77,231,0}; //grille, 00e74d
const pixel olive = {86,159,165}; //spoiler, a59f56
const pixel light_blue = {255,215,93}; //hood, 5dd7ff
const pixel dark_brown = {33,67,101}; //roof, 654321
const pixel dark_blue = {86,52,18}; //front bumper, 123456
const pixel dark_yellow = {54,206,241}; //back bumper, f1ce36
const pixel dark_red = {0,0,153}; //door handles, 990000
const pixel beige = {204,204,255}; //trunk, ffcccc
const pixel turquoise = {148,152,73}; //antenna, 499894
const pixel dark_purple = {130,0,75}; //window frames, 4b0082

const vector<pixel> labeled_colors = {blue,green,red,cyan,yellow,magenta,orange,light_green,
                                      brown,pastel_red,light_purple,light_blue,olive,purple,bright_green,dark_brown,dark_blue,
                                      dark_yellow, dark_red, dark_purple, beige, turquoise};

struct label {
    pixel color;
    string name;
};

//Labels list (must have the same color order as labeled_color)
const label mirror = {blue,"mirror"};
const label  rail = {green,"rail"};
const label wheel = {red,"wheel"};
const label frontdoor = {yellow,"frontdoor"};
const label backdoor = {cyan,"backdoor"};
const label windows = {magenta,"windows"};
const label lights = {orange,"lights"};
const label plate = {purple,"license plate"};
const label front_side = {light_green,"front side"};
const label back_side = {brown,"back side"};
const label car_front = {pastel_red,"car_front"};
const label car_back = {light_purple,"car_back"};
const label grille = {bright_green,"grille"};
const label spoiler = {olive,"spoiler"};
const label hood = {light_blue,"hood"};
const label roof = {dark_brown,"roof"};
const label front_bumper = {dark_blue,"front bumper"};
const label back_bumper = {dark_yellow,"back bumper"};
const label door_handle = {dark_red,"door handle"};
const label trunk = {beige,"trunk"};
const label antenna = {turquoise,"antenna"};
const label frames = {dark_purple,"frames"};

const vector <label> labels = {mirror,rail,wheel,frontdoor,backdoor,windows,lights,plate,front_side,back_side,
                               car_front,car_back,grille,spoiler,hood,roof,front_bumper,back_bumper,door_handle,
                               trunk,antenna,frames};

struct point{
    int x;
    int y;
};

struct aug_point{
    point coords;
    pixel color;
};

//Utility functions:
//general function to check any color in the image
bool check_pixel_bgr(const pixel& pix_img,const pixel& rgb_ref);
//flip rgb to bgr or bgr to rgb
pixel rgb_to_bgr (const pixel& rgb_values);
//hex to dec converter
int HexToDec(const string& num);
//hex to bgr converter (use capital letters for hex)
pixel hex_to_bgr(const string& hex);

//verify path and open image
Mat open_image(const string& path);
//display image
void show_image(const Mat& image);

//change file names in a folder (_l.png -> .png)
void change_names(const string & input_folder_path,  const string & output_folder_path);

//Resize image (single image)
void square_and_write(const string & output_path,  Mat & image, const int & length);
//Resize jpg and png to given size in input_folder (multiple images, jpg and png)
void resize_all_images(const string & input_folder_path,  const string & output_folder_path, const int & resize_length);


//Change specific colors  to one color (single image)
Mat modify(const Mat& image,const vector<pixel>& old_colors, const pixel& new_color);
//Change  specific colors to one color (multiple images, only png)
void modify_all(const string & input_folder_path,const string & output_folder_path,const vector<pixel> & colors, const pixel & new_color);


//find outline of specific color (single image)
Mat find_outline(const Mat& image, const vector<pixel>& colors);
//extract outline points coordinates to output file (single image)
void write_outlines_coord(const Mat & img, const vector<pixel> & colors, const string & output_file_path);

//Add outlines to original image
Mat outlines_on_original (const Mat& image, const Mat & labeled, const vector <pixel> colors);
//Add outlines to original images (multiple images, saving to png)
void outlines_on_original_all (const string & input_folder_path,  const string & output_folder_path, const vector<pixel>& colors);


//change the transparency of the whole picture with alpha blending + opencv  magic? -> Ask Audrius (single image)
Mat transparency_global(const Mat& image,const float& alpha);
//change the transparency with alpha blending (multiple images, only png)
void transparency_global_all(const string & input_folder_path,  const string & output_folder_path,const float & alpha);

//Add an alpha channel and create png transparency (single image)
 Mat transparency_global_alpha(const Mat& image,  const float& alpha);
 //change the transparency of the whole picture with alpha channel (multiple images, only png)
 void transparency_global_alpha_all(const string & input_folder_path,  const string & output_folder_path,const float & alpha);

//change the transparency of certain colors with alpha channel (single images)
Mat transparency_local_alpha(const Mat&image, const vector<pixel>& colors,const float& alpha);
//change the transparency of certain colors with alpha channel (multiple images)
void transparency_local_alpha_all(const string & input_folder_path,  const string & output_folder_path, const vector<pixel>& colors,const float& alpha);

//add transparency masks on original image (single image)
Mat  transparency_on_original(const Mat & img,const Mat & lab,const vector<pixel> & colors,const float & alpha);
//add transparency masks on original image (multiple images, output png)
void transparency_masks_original_all (const string & input_folder_path,  const string & output_folder_path, const vector<pixel>& colors,const float & alpha);


//extract masks of specific colors to black image(single image)
Mat extract(const Mat& image, const vector<pixel>& colors);
//extract masks of specific colors to black image (multiple images, only png)
void extract_all(const string & input_folder_path,const string & output_folder_path,const vector<pixel> & old_colors);


//place masks of specific color on original image (single image)
Mat masks_on_original(const Mat & img, const Mat & lab, const vector <pixel> & colors);
//place masks of specific color on original image (multiple images, output is png)
void extracted_masks_on_original_all (const string & input_folder_path,  const string & output_folder_path, const vector<pixel>& colors);


//check if the 22 details are present in an image (single image)
bool check_labeled_colors(const std::string & img_path, const cv::Mat & image, const bool & mode);
//check for labeled colors presence in jpg images (multiple images, only jpg)
void check_non_labeled_images(const string & input_folder_path);
//check for labeled colors presence in png images (multiples images, only png)
void check_labeled_images(const string & input_folder_path);


//Gaussian blur on specific colors changing on original image(single image)
Mat gaussian_blur (const Mat & img, const Mat & lab, const vector<pixel> colors,const int & ker_size);
//Gaussian blur on specific colors, changing on original image (multiple images, jpg only)
void gaussian_blur_all (const string & input_folder_path,  const string & output_folder_path, const vector<pixel> colors,const int & ker_size);

//make the whole image blurred (single image)
Mat gaussian_blur_global (const Mat & img, const int & ker_size);
//Gaussian blur on all the original image (multiple images, only jpg)
void gaussian_blur_global_all(const string & input_folder_path,  const string & output_folder_path,const int & ker_size);
//Gaussian blur on all images (multiple images, jpg and png), used for augmentation
void gaussian_blur_global_all_2(const string & input_folder_path,  const string & output_folder_path,const int & ker_size);


//Gaussian noise on original image (single image)
Mat gaussian_noise(const Mat & img, const float & mean,const float & std);
//Gaussian noise on original image (multiple images, jpg only)
void gaussian_noise_all(const string & input_folder_path,  const string & output_folder_path,const float & mean,const float & std);
//Gaussian noise on all images (multiple images, jpg and png), used for augmentation
void gaussian_noise_all_2 (const string & input_folder_path,  const string & output_folder_path,const float & mean,const float & std);

//Gaussian noise on specific colors (single image)
Mat gaussian_noise_color(const Mat & img, const vector <pixel> colors,const float & mean,const float & std);
//Gaussian noise on specific colors (multiple images, only png)
void gaussian_noise_color_all(const string & input_folder_path,  const string & output_folder_path,const vector <pixel> colors, const float & mean,const float & std);


//test functions
//jpg outline white appearance
Mat transparency_shapes(const Mat&image, const float& alpha);
//change the transparency of certain colors (doesn't work changes color)
Mat transparency_local(const Mat& image, const vector<pixel>& colors, const float& alpha);
//change the transparency of certain colors (doesn't work changes color)
Mat local_transparency(const Mat& image, const Mat& transparent_image, const vector<pixel>& colors, const float& alpha);

#endif //MASK_TOOLS_H
