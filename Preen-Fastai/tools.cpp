#include <tools.h>
#include <iostream>
#include <filesystem>
#include <fstream>

using namespace std;
using namespace cv;
using recursive_directory_iterator = std::filesystem::recursive_directory_iterator;

static bool missing_color = false;


bool check_pixel_bgr(const pixel& pix_img, const pixel& rgb_ref){

    if ((pix_img.b == rgb_ref.b)&&(pix_img.g == rgb_ref.g)&&(pix_img.r== rgb_ref.r)){
        return true;
    }
    else return false;
}

pixel rgb_to_bgr (const pixel& rgb_values){
    pixel fliped_values = {rgb_values.r,rgb_values.g,rgb_values.b};
    return fliped_values;
}

int HexToDec(const string& num)
{
    int len = num.length();
    int base = 1;
    int temp = 0;
    for (int i = len - 1; i >= 0; i--)
    {
        if (num[i] >= '0' && num[i] <= '9')
        {
            temp += (num[i] - 48) * base;
            base = base * 16;
        }
        else if (num[i] >= 'A' && num[i] <= 'F')
        {
            temp += (num[i] - 55) * base;
            base = base * 16;
        }
    }
    return temp;
}

pixel hex_to_bgr(const string& hex){

    string r = hex.substr(0,2);
    string g = hex.substr(2,2);
    string b = hex.substr(4,2);
    cout << r << endl;
    uchar r_val = HexToDec(r);
    uchar g_val = HexToDec(g);
    uchar b_val = HexToDec(b);
    cout << r_val << endl;
    pixel bgr = {b_val,g_val,r_val};
    return bgr;
}

Mat open_image(const string& path){
    Mat img = imread(path, IMREAD_COLOR);//jpg dosen't have transparency, if png use IMREAD_UNCHANGED
    if(img.empty())
    {
        cout << "Could not read the image at path: " << path << " in open_image" << endl;
        cout << "Terminating program" << endl;
        exit(EXIT_FAILURE);
    }
    return img;
}

void show_image(const Mat& image){
    imshow("Display window", image);
    waitKey(0); // Wait for a keystroke in the window
}

void change_names(const string & input_folder_path,  const string & output_folder_path){
    int n = 0;
    int f = 0;
    for (const auto &dirEntry : recursive_directory_iterator(input_folder_path)) {
        auto path = dirEntry.path();
        if (path.extension() == ".png") {
            auto dir = path.parent_path();
            auto stem = path.stem();
            auto base = dir.string() + "/" + stem.string();
            auto png = base + ".png";
            if (std::filesystem::exists(png)) {
                cv::Mat img = cv::imread(png);
                if (img.empty()) {
                    std::cout << "Could not read the image in \"change_names\"" << png << std::endl;
                    f++;
                    continue;
                }
                else {
                    string a = string(stem);
                    a.erase(a.length()-2);
                    imwrite(output_folder_path+"/"+a+".png",img);
                }
            }
            else cout << "Unable to open " << png << " in \"change_names\"" << endl;
        }
    }
}

void square_and_write(const string & output_path, Mat & image, const int & length) {
    const cv::Size size(length, length);

    cv::Mat squared;
    cv::resize(image, squared, size);
    cv::imwrite(output_path, squared);
}

void resize_all_images(const string & input_folder_path,  const string & output_folder_path, const int & resize_length){
    int n = 0;
    int f = 0;

    for (const auto &dirEntry : recursive_directory_iterator(input_folder_path)) {
        auto path = dirEntry.path();

        if (path.extension() == ".jpg") {

            auto dir = path.parent_path();
            auto stem = path.stem();
            auto base = dir.string() + "/" + stem.string();
            auto jpg = base + ".jpg";

            if (std::filesystem::exists(jpg)) {
                cv::Mat img = cv::imread(jpg);

                if (img.empty()) {
                    std::cout << "Could not read the image" << jpg << " in \" resize_all_images\"" <<std::endl;
                    f++;
                    continue;
                }
                else {
                    square_and_write(output_folder_path+"/"+string(stem)+".jpg",img,resize_length);
                    n++;
                }
            }
            else cout << "Unable to open " << jpg << " in \"resize_all_images\"" << endl;
        }
        else {
            if  (path.extension() == ".png") {
                auto dir = path.parent_path();
                auto stem = path.stem();
                auto base = dir.string() + "/" + stem.string();
                auto png = base + ".png";
                if (std::filesystem::exists(png)) {
                    cv::Mat img = cv::imread(png);
                    if (img.empty()) {
                        std::cout << "Could not read the image" << png << "in \" resize_all_images\""<< std::endl;
                        f++;
                        continue;
                    }
                    else {
                        square_and_write(output_folder_path+"/"+string(stem)+".png",img,resize_length);
                        n++;
                    }
                }
                else cout << "Unable to open " << png << " in \"resize_all_images\"" << endl;
            }
        }
    }
    printf("Resize status: %d entries successful, %d wrong\n", n, f);
}

Mat modify(const Mat& image, const vector<pixel>& old_colors, const pixel& new_color){
    if (old_colors.empty()){
        cout << "Color vector is empty, cannot perform modifcation in function \"modify\" " << endl;
        cout << "Terminating program" << endl;
        exit(EXIT_FAILURE);
    }
    //output image
    Mat modified_img = image.clone();
    missing_color = false;
    for (int k = 0; k < old_colors.size(); ++k) {
        bool found_match = false;
        for (int i = 0; i < image.rows; ++i) {
            for (int j = 0; j < image.cols; ++j) {
                pixel img_pixel = {image.at<Vec3b>(i, j)[0], image.at<Vec3b>(i, j)[1], image.at<Vec3b>(i, j)[2]};
                if (check_pixel_bgr(img_pixel, old_colors[k])) {
                    found_match = true;
                    modified_img.at<Vec3b>(i, j)[BLUE_POS] = unsigned(new_color.b);
                    modified_img.at<Vec3b>(i, j)[GREEN_POS] = unsigned(new_color.g);
                    modified_img.at<Vec3b>(i, j)[RED_POS] = unsigned(new_color.r);
                }
            }
        }
        if (!found_match){
            cout << "Color to modify: [" <<  unsigned(old_colors[k].b) << ", " << unsigned(old_colors[k].g) << ", " << unsigned(old_colors[k].r) << "]  (BGR format) is not present in given image in function: \"modify\" " << endl;
            missing_color = true;
        }
    }

    return modified_img;
}

void modify_all(const string & input_folder_path,const string & output_folder_path,const vector<pixel> & old_colors, const pixel & new_color){

    if (old_colors.empty()){
        cout << "Color vector is empty, cannot perform color modification search in function \"modify_all\" " << endl;
        cout << "Terminating program" << endl;
        exit(1);
    }
    int n = 0;
    int f = 0;
    for (const auto &dirEntry : recursive_directory_iterator(input_folder_path)) {
        auto path = dirEntry.path();
        if (path.extension() == ".png") {
            auto dir = path.parent_path();
            auto stem = path.stem();
            auto base = dir.string() + "/" + stem.string();
            auto png = base + ".png";
            if (std::filesystem::exists(png)) {
                cv::Mat img = cv::imread(png);
                if (img.empty()) {
                    std::cout << "Could not read the image" << png << " in \"modify_all\""<< std::endl;
                    f++;
                    continue;
                }
                else {
                    n++;
                    Mat modified_img = modify(img,old_colors,new_color);
                    if (missing_color){
                        cout << "Image path: " << output_folder_path+"/"+string(stem)+".png" << endl << endl;
                    }
                    imwrite(output_folder_path+"/"+string(stem)+".png",modified_img);
                }
            }
            else cout << "Unable to open " << png << " in \"modify_all\"" << endl;
        }
    }
    printf("Modify_all status: %d entries successful, %d wrong\n", n, f);
}

bool check_neighbors(const Mat& image,const int& i,const int& j, const pixel& ref){

    for (int a = -1;a < 2;++a){
        for (int b = -1; b <2 ; ++b){
            if ((a == 0)&&(b == 0)){
                continue;
            }
            pixel img_pixel = {image.at<Vec3b>(i+a, j+b)[BLUE_POS], image.at<Vec3b>(i+a, j+b)[GREEN_POS],
                               image.at<Vec3b>(i+a, j+b)[RED_POS]};
            if (!check_pixel_bgr(img_pixel,ref)){
                return false;
            }

        }
    }
    return true;
}

Mat find_outline(const Mat& image, const vector<pixel>& colors){
    //Matrix with point coordinates -> ouline to add
    if (colors.empty()){
        cout << "Color vector is empty, cannot perform outline search in function \"find_outline\" " << endl;
        cout << "Terminating program" << endl;
        exit(EXIT_FAILURE);
    }
    //creating black image
    Mat extracted_outlines(image.rows, image.cols, CV_8UC3, Scalar(0, 0, 0));

    for (int k=0; k < colors.size(); ++k){
        bool found_match = false;
        for (int i=0; i < image.rows; ++i){
            for(int j=0; j < image.cols; ++j){
                pixel img_pixel = {image.at<Vec3b>(i, j)[BLUE_POS], image.at<Vec3b>(i, j)[GREEN_POS],
                                   image.at<Vec3b>(i, j)[RED_POS]};
                if (check_pixel_bgr(img_pixel, colors[k])) {
                    found_match = true;
                    if ((i == 0)||(i == image.rows-1)||(j == 0)||(j == image.cols -1)){
                        extracted_outlines.at<Vec3b>(i, j)[BLUE_POS] = unsigned(img_pixel.b);
                        extracted_outlines.at<Vec3b>(i, j)[GREEN_POS] = unsigned(img_pixel.g);
                        extracted_outlines.at<Vec3b>(i, j)[RED_POS] = unsigned(img_pixel.r);
                    }
                    else {
                        if(!check_neighbors(image,i,j,colors[k])){
                            extracted_outlines.at<Vec3b>(i, j)[BLUE_POS] = unsigned(img_pixel.b);
                            extracted_outlines.at<Vec3b>(i, j)[GREEN_POS] = unsigned(img_pixel.g);
                            extracted_outlines.at<Vec3b>(i, j)[RED_POS] = unsigned(img_pixel.r);
                        }
                    }

                }

            }
        }
        if (!found_match){
            cout << "Color to outline : [" << unsigned(colors[k].b) << ", " << unsigned(colors[k].g) << ", " << unsigned(colors[k].r) << "]  (BGR format) is not present in given image in function: \"find_outline\""<< endl;
        }
    }

    return extracted_outlines;
}

void write_outlines_coord(const Mat & image, const vector<pixel> & colors, const string & output_file_path){
    if (colors.empty()){
        cout << "Color vector is empty, cannot perform outline search in function \"find_outline\" " << endl;
        cout << "Terminating program" << endl;
        exit(EXIT_FAILURE);
    }
    vector <point> outline_points;
    ofstream myfile (output_file_path);

    for (int k=0; k < colors.size(); ++k){
        bool found_match = false;
        for (int i=0; i < image.rows; ++i){
            for(int j=0; j < image.cols; ++j){
                pixel img_pixel = {image.at<Vec3b>(i, j)[BLUE_POS], image.at<Vec3b>(i, j)[GREEN_POS],
                                   image.at<Vec3b>(i, j)[RED_POS]};
                if (check_pixel_bgr(img_pixel, colors[k])) {
                    found_match = true;
                    if ((i == 0)||(i == image.rows-1)||(j == 0)||(j == image.cols -1)){
                        point outline_coord = {i,j};
                        outline_points.push_back(outline_coord);
                    }
                    else {
                        if(!check_neighbors(image,i,j,colors[k])){
                            point outline_coord = {i,j};
                            outline_points.push_back(outline_coord);
                        }
                    }

                }

            }
        }
        if (myfile.is_open())
        {
            myfile << "Outline points of [" << unsigned(colors[k].b) << "," << unsigned(colors[k].g) << "," << unsigned(colors[k].r) << "] (BGR)\n";
            for (int i = 0; i < outline_points.size(); ++i){
                myfile << "(" << outline_points[i].x << "," << outline_points[i].y << ")\n";
            }


        }
        else cout << "Unable to open  in \"write_outlines_coord\"";
        if (!found_match){
            cout << "Color to outline : [" << unsigned(colors[k].b) << ", " << unsigned(colors[k].g) << ", " << unsigned(colors[k].r) << "]  (BGR format) is not present in given image in function: \"find_outline\""<< endl;
        }
    }
    myfile.close();
}

vector <aug_point> recover_outlines(const Mat & image, const vector<pixel> & colors){
    vector<aug_point> outline_points;
    missing_color = false;
    for (int k = 0; k < colors.size(); ++k){
        bool found_match = false;
        for (int i = 0; i < image.rows; ++i){
            for (int j = 0; j < image.cols; ++j){
                pixel img_pixel = {image.at<Vec3b>(i, j)[BLUE_POS], image.at<Vec3b>(i, j)[GREEN_POS],
                                   image.at<Vec3b>(i, j)[RED_POS]};
                if (check_pixel_bgr(img_pixel, colors[k])) {
                    found_match = true;
                    if ((i == 0)||(i == image.rows-1)||(j == 0)||(j == image.cols -1)){
                        point coords = {i,j};
                        aug_point outline_coord = {coords,colors[k]};
                        outline_points.push_back(outline_coord);
                    }
                    else {
                        if(!check_neighbors(image,i,j,colors[k])){
                            point coords = {i,j};
                            aug_point outline_coord = {coords,colors[k]};
                            outline_points.push_back(outline_coord);
                        }
                    }
                }
            }
        }
        if (!found_match){
            missing_color = true;
            cout << "Color outline to extract : [" << unsigned(colors[k].b) << ", " << unsigned(colors[k].g) << ", " << unsigned(colors[k].r) << "]  (BGR format) is not present in given image in function: \"recover_outline\""<< endl;
        }
    }
    return outline_points;
}

Mat outlines_on_original (const Mat& image, const Mat & labeled, const vector <pixel> colors){
    if (colors.empty()){
        cout << "Color vector is empty, cannot perform outline search in function \"outlines_on_original\" " << endl;
        cout << "Terminating program" << endl;
        exit(1);
    }
    Mat original_modified = image.clone();
    vector<aug_point> outlines = recover_outlines(labeled,colors);

    if (outlines.empty()){
        cout << "Outline finder return vector is empy in \"outlines_on_oringial\"" << endl;
        cout << "Terminating program" << endl;
        exit(1);
    }
    for (int k = 0; k < outlines.size(); ++k){
        int i = outlines[k].coords.x;
        int j = outlines[k].coords.y;
        original_modified.at<Vec3b>(i, j)[BLUE_POS] = unsigned(outlines[k].color.b);
        original_modified.at<Vec3b>(i, j)[GREEN_POS] = unsigned(outlines[k].color.g);
        original_modified.at<Vec3b>(i, j)[RED_POS] = unsigned(outlines[k].color.r);
    }
    return original_modified;
}

void outlines_on_original_all (const string & input_folder_path,  const string & output_folder_path, const vector<pixel>& colors){
    if (colors.empty()){
        cout << "Color vector is empty, cannot perform outline search in function \" outlines_on_original_all\" " << endl;
        cout << "Terminating program" << endl;
        exit(1);
    }
    int n = 0;
    int f = 0;
    for (const auto &dirEntry : recursive_directory_iterator(input_folder_path)) {
        auto path = dirEntry.path();
        if (path.extension() == ".jpg") {
            auto dir = path.parent_path();
            auto stem = path.stem();
            auto base = dir.string() + "/" + stem.string();
            auto jpg = base + ".jpg";
            auto png = base + ".png";
            if (std::filesystem::exists(jpg)) {
                cv::Mat img = cv::imread(jpg);
                if (img.empty()) {
                    std::cout << "Could not read the image in \"outlines_on_original_all\"" << jpg << std::endl;
                    f++;
                    continue;
                }
                else {
                    n++;
                    if (std::filesystem::exists(png)) {
                        cv::Mat lab = cv::imread(png);
                        if (lab.empty()) {
                            std::cout << "Could not read the image in \"outlines_on_original_all\"" << png << std::endl;
                            f++;
                            continue;
                        }
                        else {
                            Mat outline_on_img = outlines_on_original(img,lab,colors);
                            if (missing_color){
                                cout << "Image path: " << png << endl;
                            }
                            imwrite(output_folder_path+"/"+string(stem)+"_outline.png",outline_on_img);
                        }
                    }
                    else cout << "Unable to open " << png << " in \"outlines_on_original_all\"" << endl;
                }
            }
            else cout << "Unable to open " << jpg << " in \"outlines_on_original_all\"" << endl;
        }
    }
    printf("outlines_on_original_all status: %d entries successful, %d wrong\n", n, f);
}

//Using the following formula: opacity*original + (1-opacity)*background = resulting pixel
//Where opacity is: 0 < alpha < 1
//There is no alpha channel involved
Mat transparency_global(const Mat& image,const float& alpha){
    if ((alpha<0)||(alpha>1)){
        cout << "Alpha value is not valid: " << alpha << ", must be 0 < alpha < 1, in function \"transparency_global\" " << endl;
        cout << "Terminating program" << endl;
        exit(EXIT_FAILURE);
    }

    Mat transparent_image = image.clone();

    //opacity*original + (1-opacity)*background
    for (int i = 0; i < image.rows; ++i){
        for (int j = 0; j < image.cols; ++j){
            pixel img_pixel = {image.at<Vec3b>(i, j)[BLUE_POS], image.at<Vec3b>(i, j)[GREEN_POS],
                               image.at<Vec3b>(i, j)[RED_POS]};

            img_pixel.b = uchar(alpha*img_pixel.b + (1-alpha)*WHITE_BACKGROUND);
            img_pixel.g = uchar(alpha*img_pixel.g + (1-alpha)*WHITE_BACKGROUND);
            img_pixel.r = uchar(alpha*img_pixel.r + (1-alpha)*WHITE_BACKGROUND);
            //cout << "Global" << unsigned(img_pixel.b) << " " << unsigned(img_pixel.g) << " " << unsigned(img_pixel.r) << endl;
            transparent_image.at<Vec3b>(i, j)[BLUE_POS] = unsigned(img_pixel.b);
            transparent_image.at<Vec3b>(i, j)[GREEN_POS] = unsigned(img_pixel.g);
            transparent_image.at<Vec3b>(i, j)[RED_POS] = unsigned(img_pixel.r);

        }
    }

    return transparent_image;
}

void transparency_global_all(const string & input_folder_path,  const string & output_folder_path,const float & alpha){
    if ((alpha<0)||(alpha>1)){
        cout << "Alpha value is not valid: " << alpha << ", must be 0 < alpha < 1, in function \"transparency_global_all\" " << endl;
        cout << "Terminating program" << endl;
        exit(1);
    }

    int n = 0;
    int f = 0;
    for (const auto &dirEntry : recursive_directory_iterator(input_folder_path)) {
        auto path = dirEntry.path();
        if (path.extension() == ".png") {
            auto dir = path.parent_path();
            auto stem = path.stem();
            auto base = dir.string() + "/" + stem.string();
            auto png = base + ".png";
            if (std::filesystem::exists(png)) {
                cv::Mat img = cv::imread(png);
                if (img.empty()) {
                    std::cout << "Could not read the image in \"transparency_global_all\"" << png << std::endl;
                    f++;
                    continue;
                }
                else {
                    n++;
                    Mat transparent_img = transparency_global(img,alpha);
                    imwrite(output_folder_path+"/"+string(stem)+".png",transparent_img);
                }
            }
            else cout << "Unable to open " << png << " in \"transparency_global_all\"" << endl;
        }
    }
    printf("Transparency_global_all status: %d entries successful, %d wrong\n", n, f);
}

Mat transparency_global_alpha(const Mat& image, const float& alpha){
    if ((alpha<0)||(alpha>1)){
        cout << "Alpha value is not valid: " << alpha << ", must be 0 < alpha < 1, in function \"transparency_global_alpha\" " << endl;
        cout << "Terminating program" << endl;
        exit(EXIT_FAILURE);
    }

    Mat image_bgra;
    cvtColor(image,image_bgra,COLOR_BGR2BGRA);

    Mat bgra[4];
    split(image_bgra,bgra);

    bgra[ALPHA_POS] = bgra[ALPHA_POS] * alpha;
    merge(bgra,4,image_bgra);

    return image_bgra;
}

void transparency_global_alpha_all(const string & input_folder_path,  const string & output_folder_path,const float & alpha){
    if ((alpha<0)||(alpha>1)){
        cout << "Alpha value is not valid: " << alpha << ", must be 0 < alpha < 1, in function \"transparency_global_alpha_all\" " << endl;
        cout << "Terminating program" << endl;
        exit(1);
    }


    int n = 0;
    int f = 0;
    for (const auto &dirEntry : recursive_directory_iterator(input_folder_path)) {
        auto path = dirEntry.path();
        if (path.extension() == ".png") {
            auto dir = path.parent_path();
            auto stem = path.stem();
            auto base = dir.string() + "/" + stem.string();
            auto png = base + ".png";
            if (std::filesystem::exists(png)) {
                cv::Mat img = cv::imread(png);
                if (img.empty()) {
                    std::cout << "Could not read the image in \"transparency_global_alpha_all\"" << png << std::endl;
                    f++;
                    continue;
                }
                else {
                    n++;
                    Mat transparent_img = transparency_global_alpha(img, alpha);
                    imwrite(output_folder_path+"/"+string(stem)+".png",transparent_img);
                }
            }
            else cout << "Unable to open " << png << " in \"transparency_global_alpha_all\"" << endl;
        }
    }
    printf("Transparency_global_alpha_all status: %d entries successful, %d wrong\n", n, f);
}

Mat transparency_local_alpha(const Mat&image, const vector<pixel>& colors,const float& alpha){
    if ((alpha<0)||(alpha>1)){
        cout << "Alpha value is not valid: " << alpha << ", must be 0 < alpha < 1, in function \"transparency_local_alpha\" " << endl;
        cout << "Terminating program" << endl;
        exit(EXIT_FAILURE);
    }

    if (colors.empty()){
        cout << "Color vector is empty, cannot perform outline search in function \"transparency_local_alpha\" " << endl;
        cout << "Terminating program" << endl;
        exit(EXIT_FAILURE);
    }
    missing_color = false;
    Mat image_bgra;
    cvtColor(image,image_bgra,COLOR_BGR2BGRA);

    Mat bgra[4];
    split(image_bgra,bgra);

    for (int k = 0; k < colors.size(); ++k) {

        bool found_match = false;

        for (int i = 0; i < image.rows; ++i) {
            for (int j = 0; j < image.cols; ++j) {
                pixel img_pixel = {image.at<Vec3b>(i, j)[BLUE_POS], image.at<Vec3b>(i, j)[GREEN_POS],
                                   image.at<Vec3b>(i, j)[RED_POS]};
                if (check_pixel_bgr(img_pixel, colors[k])) {
                    found_match = true;
                    bgra[ALPHA_POS].at<uchar>(i, j) = bgra[ALPHA_POS].at<uchar>(i, j)*alpha;
                }
            }
        }
        if (!found_match){
            missing_color = true;
            cout << "Color to replace: [" << unsigned(colors[k].b) << ", " << unsigned(colors[k].g) << ", " << unsigned(colors[k].r)<< "]  (BGR format) is not present in given image in function: \"transparency_local_alpha\""<< endl;
        }
    }

    merge(bgra,4,image_bgra);

    return image_bgra;

}

void transparency_local_alpha_all(const string & input_folder_path,  const string & output_folder_path, const vector<pixel>& colors,const float& alpha){
    if ((alpha<0)||(alpha>1)){
        cout << "Alpha value is not valid: " << alpha << ", must be 0 < alpha < 1, in function \"transparency_local_alpha_all\" " << endl;
        cout << "Terminating program" << endl;
        exit(EXIT_FAILURE);
    }

    if (colors.empty()){
        cout << "Color vector is empty, cannot perform transparency change in function \"transparency_local_alpha_all\" " << endl;
        cout << "Terminating program" << endl;
        exit(1);
    }
    int n = 0;
    int f = 0;
    for (const auto &dirEntry : recursive_directory_iterator(input_folder_path)) {
        auto path = dirEntry.path();
        if (path.extension() == ".png") {
            auto dir = path.parent_path();
            auto stem = path.stem();
            auto base = dir.string() + "/" + stem.string();
            auto png = base + ".png";
            if (std::filesystem::exists(png)) {
                cv::Mat img = cv::imread(png);
                if (img.empty()) {
                    std::cout << "Could not read the image in \"transparency_local_alpha_all\"" << png << std::endl;
                    f++;
                    continue;
                }
                else {
                    n++;
                    Mat transparent_img = transparency_local_alpha(img,colors,alpha);
                    if (missing_color){
                        cout << "Image path: " << output_folder_path+"/"+string(stem)+".png" << endl << endl;
                    }
                    imwrite(output_folder_path+"/"+string(stem)+".png",transparent_img);
                }
            }
            else cout << "Unable to open " << png << " in \"transparency_local_alpha_all\"" << endl;
        }
    }
    printf("Transparency_local_alpha_all status: %d entries successful, %d wrong\n", n, f);
}

Mat  transparency_on_original(const Mat & img,const Mat & lab,const vector<pixel> & colors,const float & alpha){
    if ((alpha<0)||(alpha>1)){
        cout << "Alpha value is not valid: " << alpha << ", must be 0 < alpha < 1, in function \"transparency_on_original\" " << endl;
        cout << "Terminating program" << endl;
        exit(EXIT_FAILURE);
    }

    if (colors.empty()){
        cout << "Color vector is empty, cannot perform transparency change in function \"transparency_on_original\" " << endl;
        cout << "Terminating program" << endl;
        exit(1);
    }

    missing_color = false;

    Mat transparent_img;
    cvtColor(img,transparent_img,COLOR_BGR2BGRA);

    Mat bgra[4];
    split(transparent_img,bgra);

    for (int k = 0; k < colors.size(); ++k) {

        bool found_match = false;

        for (int i = 0; i < img.rows; ++i) {
            for (int j = 0; j < img.cols; ++j) {
                pixel img_pixel = {lab.at<Vec3b>(i, j)[BLUE_POS], lab.at<Vec3b>(i, j)[GREEN_POS],
                                   lab.at<Vec3b>(i, j)[RED_POS]};
                if (check_pixel_bgr(img_pixel, colors[k])) {
                    found_match = true;
                    bgra[BLUE_POS].at<uchar>(i, j) = unsigned(img_pixel.b);
                    bgra[GREEN_POS].at<uchar>(i, j) = unsigned(img_pixel.g);
                    bgra[RED_POS].at<uchar>(i, j) = unsigned(img_pixel.r);
                    bgra[ALPHA_POS].at<uchar>(i, j) = bgra[ALPHA_POS].at<uchar>(i, j)*alpha;

                }
            }
        }
        if (!found_match){
            missing_color = true;
            cout << "Color to replace: [" << unsigned(colors[k].b) << ", " << unsigned(colors[k].g) << ", " << unsigned(colors[k].r)<< "]  (BGR format) is not present in given image in function: \"transparency_on_original\""<< endl;
        }
    }


    merge(bgra,4,transparent_img);

    return transparent_img;

}

void transparency_masks_original_all (const string & input_folder_path,  const string & output_folder_path, const vector<pixel>& colors,const float & alpha){
    if (colors.empty()){
        cout << "Color vector is empty, cannot perform outline search in function \" transparency_masks_original_all\" " << endl;
        cout << "Terminating program" << endl;
        exit(1);
    }
    int n = 0;
    int f = 0;
    for (const auto &dirEntry : recursive_directory_iterator(input_folder_path)) {
        auto path = dirEntry.path();
        if (path.extension() == ".jpg") {
            auto dir = path.parent_path();
            auto stem = path.stem();
            auto base = dir.string() + "/" + stem.string();
            auto jpg = base + ".jpg";
            auto png = base + ".png";
            if (std::filesystem::exists(jpg)) {
                cv::Mat img = cv::imread(jpg);
                if (img.empty()) {
                    std::cout << "Could not read the image in \"transparency_masks_original_all\"" << jpg << std::endl;
                    f++;
                    continue;
                }
                else {
                    n++;
                    if (std::filesystem::exists(png)) {
                        cv::Mat lab = cv::imread(png);
                        if (lab.empty()) {
                            std::cout << "Could not read the image in \"transparency_masks_original_all\"" << png << std::endl;
                            f++;
                            continue;
                        }
                        else {
                            Mat transparency_on_img = transparency_on_original(img,lab,colors,alpha);
                            if (missing_color){
                                cout << "Image path: " << png << endl;
                            }
                            imwrite(output_folder_path+"/"+string(stem)+"_transparency.png",transparency_on_img);
                        }
                    }
                    else cout << "Unable to open " << png << " in \"transparency_masks_original_all\"" << endl;
                }
            }
            else cout << "Unable to open " << jpg << " in \"transparency_masks_original_all\"" << endl;
        }
    }
    printf("transparency_masks_original_all status: %d entries successful, %d wrong\n", n, f);
}

Mat extract(const Mat& image, const vector<pixel>& colors){
    if (colors.empty()){
        cout << "Color vector is empty, cannot perform extraction in function \"extract\" " << endl;
        cout << "Terminating program" << endl;
        exit(EXIT_FAILURE);
    }
    missing_color = false;
    //creating black image
    Mat extracted_masks(image.rows, image.cols, CV_8UC3, Scalar(0, 0, 0));

    for (int k = 0; k < colors.size(); ++k) {
        bool found_match = false;
        for (int i = 0; i < image.rows; ++i) {
            for (int j = 0; j < image.cols; ++j) {
                pixel img_pixel = {image.at<Vec3b>(i, j)[BLUE_POS], image.at<Vec3b>(i, j)[GREEN_POS],
                                   image.at<Vec3b>(i, j)[RED_POS]};

                if (check_pixel_bgr(img_pixel, colors[k])) {
                    found_match = true;
                    extracted_masks.at<Vec3b>(i, j)[BLUE_POS] = unsigned(img_pixel.b);
                    extracted_masks.at<Vec3b>(i, j)[GREEN_POS] = unsigned(img_pixel.g);
                    extracted_masks.at<Vec3b>(i, j)[RED_POS] = unsigned(img_pixel.r);
                }
            }
        }
        if (!found_match){
            missing_color = true;
            cout << "Color to extract: [" << unsigned(colors[k].b) << ", " << unsigned(colors[k].g) << ", " << unsigned(colors[k].r) << "]  (BGR format) is not present in given image in function: \"extract\" " << endl;
        }
    }
    return extracted_masks;
}

void extract_all(const string & input_folder_path,const string & output_folder_path,const vector<pixel> & old_colors){

    if (old_colors.empty()){
        cout << "Color vector is empty, cannot perform extract search in function \"extract_all\" " << endl;
        cout << "Terminating program" << endl;
        exit(1);
    }

    int n = 0;
    int f = 0;
    for (const auto &dirEntry : recursive_directory_iterator(input_folder_path)) {
        auto path = dirEntry.path();
        if (path.extension() == ".png") {
            auto dir = path.parent_path();
            auto stem = path.stem();
            auto base = dir.string() + "/" + stem.string();
            auto png = base + ".png";
            if (std::filesystem::exists(png)) {
                cv::Mat img = cv::imread(png);
                if (img.empty()) {
                    std::cout << "Could not read the image" << png << " in \"extract_all\"" << std::endl;
                    f++;
                    continue;
                }
                else {
                    n++;
                    Mat extracted_img = extract(img, old_colors);
                    if (missing_color){
                        cout << "Image path: " << output_folder_path+"/"+string(stem)+".png" << endl << endl;
                    }
                    imwrite(output_folder_path+"/"+string(stem)+".png",extracted_img);
                }
            }
            else cout << "Unable to open " << png << " in \"extract_all\"" << endl;
        }
    }
    printf("Extract_all status: %d entries successful, %d wrong\n", n, f);
}

//void traparent_masks_on_original_all (const string & input_folder_path,  const string & output_folder_path, const vector<pixel>& colors);
Mat masks_on_original(const Mat & img, const Mat & lab, const vector <pixel> & colors){
    if (colors.empty()){
        cout << "Color vector is empty, cannot perform outline search in function \" outlines_on_original_all\" " << endl;
        cout << "Terminating program" << endl;
        exit(1);
    }
    Mat modified_original = img.clone();
    missing_color = false;
    for (int k = 0; k < colors.size();++k){
        bool found_match = false;
        for (int i = 0; i < img.rows; ++i){
            for (int j = 0; j < img.cols; ++j){
                pixel img_pixel = {lab.at<Vec3b>(i, j)[BLUE_POS], lab.at<Vec3b>(i, j)[GREEN_POS],
                                   lab.at<Vec3b>(i, j)[RED_POS]};
                if (check_pixel_bgr(img_pixel, colors[k])) {
                    found_match = true;
                    modified_original.at<Vec3b>(i, j)[BLUE_POS] = unsigned(img_pixel.b);
                    modified_original.at<Vec3b>(i, j)[GREEN_POS] = unsigned(img_pixel.g);
                    modified_original.at<Vec3b>(i, j)[RED_POS] = unsigned(img_pixel.r);
                }
            }
        }
        if (!found_match){
            missing_color = true;
            cout << "Color to replace: [" << unsigned(colors[k].b) << ", " << unsigned(colors[k].g) << ", " << unsigned(colors[k].r)<< "]  (BGR format) is not present in given image in function: \"outlines_on_original_all\""<< endl;
        }
    }
    return modified_original;

}

void extracted_masks_on_original_all (const string & input_folder_path,  const string & output_folder_path, const vector<pixel>& colors){
    if (colors.empty()){
        cout << "Color vector is empty, cannot perform outline search in function \" extracted_masks_on_original_all\" " << endl;
        cout << "Terminating program" << endl;
        exit(1);
    }
    int n = 0;
    int f = 0;
    for (const auto &dirEntry : recursive_directory_iterator(input_folder_path)) {
        auto path = dirEntry.path();
        if (path.extension() == ".jpg") {
            auto dir = path.parent_path();
            auto stem = path.stem();
            auto base = dir.string() + "/" + stem.string();
            auto jpg = base + ".jpg";
            auto png = base + ".png";
            if (std::filesystem::exists(jpg)) {
                cv::Mat img = cv::imread(jpg);
                if (img.empty()) {
                    std::cout << "Could not read the image in \"extracted_masks_on_original_all\"" << jpg << std::endl;
                    f++;
                    continue;
                }
                else {
                    n++;
                    if (std::filesystem::exists(png)) {
                        cv::Mat lab = cv::imread(png);
                        if (lab.empty()) {
                            std::cout << "Could not read the image in \"extracted_masks_on_original_all\"" << png << std::endl;
                            f++;
                            continue;
                        }
                        else {
                            Mat masks_on_img = masks_on_original(img,lab,colors);
                            if (missing_color){
                                cout << "Image path: " << png << endl;
                            }
                            imwrite(output_folder_path+"/"+string(stem)+"_modified.png",masks_on_img);
                        }
                    }
                    else cout << "Unable to open " << png << " in \"extracted_masks_on_original_all\"" << endl;
                }
            }
            else cout << "Unable to open " << jpg << " in \"extracted_masks_on_original_all\"" << endl;
        }
    }
    printf("extracted_masks_on_original_all status: %d entries successful, %d wrong\n", n, f);
}

bool check_labeled_colors(const std::string & img_path, const cv::Mat & image, const bool & mode){
    bool found_color = false;
    vector<bool> color_found_status (labeled_colors.size(), false);
    for (int i = 0; i < image.rows; ++i) {
        for (int j = 0; j < image.cols; ++j) {
            pixel img_pixel = {image.at<cv::Vec3b>(i, j)[BLUE_POS], image.at<cv::Vec3b>(i, j)[GREEN_POS],
                               image.at<cv::Vec3b>(i, j)[RED_POS]};
            for (int k = 0; k < labeled_colors.size(); ++k){
                if (!color_found_status[k]){
                    if (check_pixel_bgr(img_pixel, labeled_colors[k])) {
                        found_color = true;
                        if ((!color_found_status[k])&&(mode == 0)){
                            color_found_status[k] = true;
                            std::cout << "Color: [" << unsigned(labeled_colors[k].b) << ", " << unsigned(labeled_colors[k].g)
                            << ", " << unsigned(labeled_colors[k].r) << "] (BGR format) which referes to \"" << labels[k].name << "\" is present in original image" << std::endl;
                        }
                        break;
                    }
                }
            }
        }
    }
    if ((found_color)&&(mode == 0)){
        std::cout << "Label color present in " << std::string(img_path) << std::endl;
        return false;
    }
    else {
        if ((!found_color)&&(mode == 0)){
            //std::cout << "No Label colors present in non " << std::string(img_path) << std::endl;
            return true;
        }
        else{
            if ((!found_color)&&(mode == 1)){
                std::cout << "No label color present in " << std::string(img_path) << std::endl;
                return false;
            }
            else{
                    //std::cout << "Label color present in " << std::string(img_path) << std::endl;
                    return true;

            }
        }
    }
}

void check_non_labeled_images(const string & input_folder_path){
    int n = 0;
    int f = 0;
    for (const auto &dirEntry : recursive_directory_iterator(input_folder_path)) {
        bool check_local = false;
        auto path = dirEntry.path();

        if (path.extension() == ".jpg") {

            auto dir = path.parent_path();
            auto stem = path.stem();
            auto base = dir.string() + "/" + stem.string();
            auto jpg = base + ".jpg";

            if (std::filesystem::exists(jpg)) {
                cv::Mat img = cv::imread(jpg);

                if (img.empty()) {
                    std::cout << "Could not read the image" << jpg << " in \"checked_non_labeled_images\"" << std::endl;
                    continue;
                }
                else {
                    check_local = check_labeled_colors(jpg, img, 0);
                    n++;
                    if (check_local == false){
                        f++;
                    }
                }
            }
            else cout << "Unable to open " << jpg << " in \"check_non_labeled_images\"" << endl;
        }
    }
    printf("Checking non labeled images status: %d entries checked, %d wrong\n", n, f);
}

void check_labeled_images(const string & input_folder_path){
    int n = 0;
    int f = 0;
    for (const auto &dirEntry : recursive_directory_iterator(input_folder_path)) {
        bool check_local = false;
        auto path = dirEntry.path();

        if (path.extension() == ".png") {

            auto dir = path.parent_path();
            auto stem = path.stem();
            auto base = dir.string() + "/" + stem.string();
            auto png = base + ".png";

            if (std::filesystem::exists(png)) {
                cv::Mat img = cv::imread(png);

                if (img.empty()) {
                    std::cout << "Could not read the image" << png << " in \"check_labeled_images\"" << std::endl;
                    continue;
                }
                else {
                    check_local = check_labeled_colors(png, img, 1);
                    n++;
                    if (check_local == false){
                        f++;
                    }
                }
            }
            else cout << "Unable to open " << png << " in \"check_labeled_images\"" << endl;
        }
    }
    printf("Checking labeled images status: %d entries checked, %d wrong\n", n, f);
}

Mat gaussian_blur (const Mat & img, const Mat & lab, const vector<pixel> colors,const int & ker_size){
    if (colors.empty()){
        cout << "Color vector is empty, cannot perform outline search in function \" gaussian_blur\" " << endl;
        cout << "Terminating program" << endl;
        exit(1);
    }
    if (ker_size <1){
        cout << "Incorrect Kernel Size, Kernel Size must be > 0" << endl;
        cout << "Terminating program" << endl;
        exit(1);
    }
    Mat gaussian_blur;
    GaussianBlur(img,gaussian_blur,Size(ker_size, ker_size),0);
    Mat blured_original = img.clone();
    missing_color = false;
    for (int k = 0; k < colors.size(); ++k){
        bool found_match = false;
        for (int i = 0; i < img.rows; ++i){
            for (int j = 0; j < img.cols; ++j){
                pixel img_pixel = {lab.at<Vec3b>(i, j)[BLUE_POS], lab.at<Vec3b>(i, j)[GREEN_POS],
                                   lab.at<Vec3b>(i, j)[RED_POS]};
                if (check_pixel_bgr(img_pixel, colors[k])) {
                    found_match = true;
                    blured_original.at<Vec3b>(i, j)[BLUE_POS] = gaussian_blur.at<Vec3b>(i, j)[BLUE_POS];
                    blured_original.at<Vec3b>(i, j)[GREEN_POS] = gaussian_blur.at<Vec3b>(i, j)[GREEN_POS];
                    blured_original.at<Vec3b>(i, j)[RED_POS] = gaussian_blur.at<Vec3b>(i, j)[RED_POS];
                }
            }
        }
        if (!found_match){
            missing_color = true;
            cout << "Color to replace: [" << unsigned(colors[k].b) << ", " << unsigned(colors[k].g) << ", " << unsigned(colors[k].r)<< "]  (BGR format) is not present in given image in function: \"gaussian_blur\""<< endl;
        }
    }
    return blured_original;
}

void gaussian_blur_all (const string & input_folder_path,  const string & output_folder_path, const vector<pixel> colors,const int & ker_size){
    if (colors.empty()){
        cout << "Color vector is empty, cannot perform outline search in function \" gaussian_blur_all\" " << endl;
        cout << "Terminating program" << endl;
        exit(1);
    }
    if (ker_size <1){
        cout << "Incorrect Kernel Size, Kernel Size must be > 0 in function \"gaussian_blur_all\"" << endl;
        cout << "Terminating program" << endl;
        exit(1);
    }

    int n = 0;
    int f = 0;
    for (const auto &dirEntry : recursive_directory_iterator(input_folder_path)) {
        auto path = dirEntry.path();
        if (path.extension() == ".jpg") {
            auto dir = path.parent_path();
            auto stem = path.stem();
            auto base = dir.string() + "/" + stem.string();
            auto jpg = base + ".jpg";
            auto png = base + ".png";
            if (std::filesystem::exists(jpg)) {
                cv::Mat img = cv::imread(jpg);
                if (img.empty()) {
                    std::cout << "Could not read the image in \"gaussian_blur_all\"" << jpg << std::endl;
                    f++;
                    continue;
                }
                else {
                    n++;
                    if (std::filesystem::exists(png)) {
                        cv::Mat lab = cv::imread(png);
                        if (lab.empty()) {
                            std::cout << "Could not read the image in \"gaussian_blur_all\"" << png << std::endl;
                            f++;
                            continue;
                        }
                        else {
                            Mat gaussian_blured = gaussian_blur(img,lab,colors,ker_size);
                            if (missing_color){
                                cout << "Image path: " << png << endl;
                            }
                            imwrite(output_folder_path+"/"+string(stem)+"_gb.jpg",gaussian_blured);
                        }
                    }
                    else cout << "Unable to open " << png << " in \"gaussian_blur_all\"" << endl;
                }
            }
            else cout << "Unable to open " << jpg << " in \"gaussian_blur_all\"" << endl;
        }
    }
    printf("gaussian_blur_all status: %d entries successful, %d wrong\n", n, f);

}

Mat gaussian_blur_global (const Mat & img, const int & ker_size){
    if (ker_size <1){
        cout << "Incorrect Kernel Size, Kernel Size must be > 0" << endl;
        cout << "Terminating program" << endl;
        exit(1);
    }
    Mat gaussian_blur;
    GaussianBlur(img,gaussian_blur,Size(ker_size, ker_size),0);
    return gaussian_blur;
}

void gaussian_blur_global_all(const string & input_folder_path,  const string & output_folder_path,const int & ker_size){

    if (ker_size <1){
        cout << "Incorrect Kernel Size, Kernel Size must be > 0 in function \"gaussian_blur_global_all\"" << endl;
        cout << "Terminating program" << endl;
        exit(1);
    }

    int n = 0;
    int f = 0;
    for (const auto &dirEntry : recursive_directory_iterator(input_folder_path)) {
        auto path = dirEntry.path();
        if (path.extension() == ".jpg") {
            auto dir = path.parent_path();
            auto stem = path.stem();
            auto base = dir.string() + "/" + stem.string();
            auto jpg = base + ".jpg";
            auto png = base + ".png";
            if (std::filesystem::exists(jpg)) {
                cv::Mat img = cv::imread(jpg);
                if (img.empty()) {
                    std::cout << "Could not read the image in \"gaussian_blur_global_all\"" << jpg << std::endl;
                    f++;
                    continue;
                }
                else {
                    n++;
                    if (std::filesystem::exists(png)) {
                        cv::Mat lab = cv::imread(png);
                        if (lab.empty()) {
                            std::cout << "Could not read the image in \"gaussian_blur_global_all\"" << png << std::endl;
                            f++;
                            continue;
                        }
                        else {
                            Mat gaussian_blured = gaussian_blur_global(img,ker_size);
                            if (missing_color){
                                cout << "Image path: " << png << endl;
                            }
                            imwrite(output_folder_path+"/"+string(stem)+"_gbg.jpg",gaussian_blured);
                        }
                    }
                    else cout << "Unable to open " << png << " in \"gaussian_blur_global_all\"" << endl;
                }
            }
            else cout << "Unable to open " << jpg << " in \"gaussian_blur_global_all\"" << endl;
        }
    }
    printf("gaussian_blur_global_all status: %d entries successful, %d wrong\n", n, f);
}

void gaussian_blur_global_all_2(const string & input_folder_path,  const string & output_folder_path,const int & ker_size){
    if (ker_size <1){
        cout << "Incorrect Kernel Size, Kernel Size must be > 0 in function \"gaussian_blur_global_all_2\"" << endl;
        cout << "Terminating program" << endl;
        exit(1);
    }


    int n = 0;
    int f = 0;
    for (const auto &dirEntry : recursive_directory_iterator(input_folder_path)) {
        auto path = dirEntry.path();

        if (path.extension() == ".jpg") {

            auto dir = path.parent_path();
            auto stem = path.stem();
            auto base = dir.string() + "/" + stem.string();
            auto jpg = base + ".jpg";

            if (std::filesystem::exists(jpg)) {
                cv::Mat img = cv::imread(jpg);

                if (img.empty()) {
                    std::cout << "Could not read the image" << jpg << " in \" gaussian_blur_global_all_2\"" <<std::endl;
                    f++;
                    continue;
                }
                else {
                    Mat blur_1 = gaussian_blur_global(img,ker_size);
                    imwrite(output_folder_path+"/"+string(stem)+"_gb.jpg",blur_1);
                    n++;
                }
            }
            else cout << "Unable to open " << jpg << " in \"gaussian_blur_global_all_2\"" << endl;
        }
        else {
            if  (path.extension() == ".png") {
                auto dir = path.parent_path();
                auto stem = path.stem();
                auto base = dir.string() + "/" + stem.string();
                auto png = base + ".png";
                if (std::filesystem::exists(png)) {
                    cv::Mat img = cv::imread(png);
                    if (img.empty()) {
                        std::cout << "Could not read the image" << png << "in \" gaussian_blur_global_all_2\""<< std::endl;
                        f++;
                        continue;
                    }
                    else {
                        Mat blur_2 = gaussian_blur_global(img,ker_size);
                        imwrite(output_folder_path+"/"+string(stem)+"_gb.png",blur_2);
                        n++;
                    }
                }
                else cout << "Unable to open " << png << " in \"gaussian_blur_global_all_2\"" << endl;
            }
        }
    }
    printf("gaussian_blur_global_all_2: %d entries successful, %d wrong\n", n, f);
}

Mat gaussian_noise(const Mat & img, const float & mean,const float & std){

    if (mean < 0){
        cout << "Incorrect mean value, mean must be >= 0, mean entered = " << mean << " ,in function \"gaussian_noise\"" << endl;
        cout << "Terminating program" << endl;
        exit(1);
    }

    if (std <0){
        cout << "Incorrect std value, mean must be >= 0 and, std entered " << std << "  ,in function \"gaussian_noise\"" << endl;
        cout << "Terminating program" << endl;
        exit(1);
    }
    Mat noise_original;

    Mat mSrc_16SC; //noise can be negative  -> using signed 16 bits to not loose precision
    Mat mGaussian_noise = Mat(img.size(),CV_16SC3);
    randn(mGaussian_noise,Scalar::all(mean), Scalar::all(std));

    img.convertTo(mSrc_16SC,CV_16SC3);
    addWeighted(mSrc_16SC, 1.0, mGaussian_noise, 1.0, 0.0, mSrc_16SC);
    mSrc_16SC.convertTo(noise_original,img.type());

    return noise_original;
}

void gaussian_noise_all(const string & input_folder_path,  const string & output_folder_path,const float & mean,const float & std){
    if (mean < 0){
        cout << "Incorrect mean value, mean must be >= 0, mean entered = " << mean << " ,in function \"gaussian_noise_all\"" << endl;
        cout << "Terminating program" << endl;
        exit(1);
    }

    if (std <0){
        cout << "Incorrect std value, mean must be >= 0 and, std entered " << std << "  ,in function \"gaussian_noise_all\"" << endl;
        cout << "Terminating program" << endl;
        exit(1);
    }

    int n = 0;
    int f = 0;
    for (const auto &dirEntry : recursive_directory_iterator(input_folder_path)) {
        auto path = dirEntry.path();
        if (path.extension() == ".jpg") {
            auto dir = path.parent_path();
            auto stem = path.stem();
            auto base = dir.string() + "/" + stem.string();
            auto jpg = base + ".jpg";
            auto png = base + ".png";
            if (std::filesystem::exists(jpg)) {
                cv::Mat img = cv::imread(jpg);
                if (img.empty()) {
                    std::cout << "Could not read the image in \"gaussian_noise_all\"" << jpg << std::endl;
                    f++;
                    continue;
                }
                else {
                    n++;
                    if (std::filesystem::exists(png)) {
                        cv::Mat lab = cv::imread(png);
                        if (lab.empty()) {
                            std::cout << "Could not read the image in \"gaussian_noise_all\"" << png << std::endl;
                            f++;
                            continue;
                        }
                        else {
                            Mat gaussian_noise_img = gaussian_noise(img,mean,std);
                            if (missing_color){
                                cout << "Image path: " << png << endl;
                            }
                            imwrite(output_folder_path+"/"+string(stem)+"_gn.jpg",gaussian_noise_img);
                        }
                    }
                    else cout << "Unable to open " << png << " in \"gaussian_noise_all\"" << endl;
                }
            }
            else cout << "Unable to open " << jpg << " in \"gaussian_noise_all\"" << endl;
        }
    }
    printf("gaussian_noise_all status: %d entries successful, %d wrong\n", n, f);

}

void gaussian_noise_all_2 (const string & input_folder_path,  const string & output_folder_path,const float & mean,const float & std){
    if (mean < 0){
        cout << "Incorrect mean value, mean must be >= 0, mean entered = " << mean << " ,in function \"gaussian_noise_all_2\"" << endl;
        cout << "Terminating program" << endl;
        exit(1);
    }

    if (std <0){
        cout << "Incorrect std value, mean must be >= 0 and, std entered " << std << "  ,in function \"gaussian_noise_all_2\"" << endl;
        cout << "Terminating program" << endl;
        exit(1);
    }

    int n = 0;
    int f = 0;
    for (const auto &dirEntry : recursive_directory_iterator(input_folder_path)) {
        auto path = dirEntry.path();

        if (path.extension() == ".jpg") {

            auto dir = path.parent_path();
            auto stem = path.stem();
            auto base = dir.string() + "/" + stem.string();
            auto jpg = base + ".jpg";

            if (std::filesystem::exists(jpg)) {
                cv::Mat img = cv::imread(jpg);

                if (img.empty()) {
                    std::cout << "Could not read the image" << jpg << " in \" gaussian_noise_all_2\"" <<std::endl;
                    f++;
                    continue;
                }
                else {
                    Mat noise_1 = gaussian_noise(img,mean,std);
                    imwrite(output_folder_path+"/"+string(stem)+"_gn.jpg",noise_1);
                    n++;
                }
            }
            else cout << "Unable to open " << jpg << " in \"gaussian_noise_all_2\"" << endl;
        }
        else {
            if  (path.extension() == ".png") {
                auto dir = path.parent_path();
                auto stem = path.stem();
                auto base = dir.string() + "/" + stem.string();
                auto png = base + ".png";
                if (std::filesystem::exists(png)) {
                    cv::Mat img = cv::imread(png);
                    if (img.empty()) {
                        std::cout << "Could not read the image" << png << "in \" gaussian_noise_all_2\""<< std::endl;
                        f++;
                        continue;
                    }
                    else {
                        Mat noise_2 = gaussian_noise(img,mean,std);
                        imwrite(output_folder_path+"/"+string(stem)+"_gn.png",noise_2);
                        n++;
                    }
                }
                else cout << "Unable to open " << png << " in \"gaussian_noise_all_2\"" << endl;
            }
        }
    }
    printf("gaussian_noise_all_2: %d entries successful, %d wrong\n", n, f);
}

Mat gaussian_noise_color(const Mat & img, const vector <pixel> colors,const float & mean,const float & std){
    if (mean < 0){
        cout << "Incorrect mean value, mean must be >= 0, mean entered = " << mean << " ,in function \"gaussian_noise_color\"" << endl;
        cout << "Terminating program" << endl;
        exit(1);
    }

    if (std <0){
        cout << "Incorrect std value, mean must be >= 0 and, std entered " << std << "  ,in function \"gaussian_noise_color\"" << endl;
        cout << "Terminating program" << endl;
        exit(1);
    }

    if (colors.empty()){
        cout << "Color vector is empty, cannot perform outline search in function \" gaussian_noise_color\" " << endl;
        cout << "Terminating program" << endl;
        exit(1);
    }

    Mat noise_original;
    Mat mSrc_16SC; //noise can be negative  -> using signed 16 bits to not loose precision
    Mat mGaussian_noise = Mat(img.size(),CV_16SC3);
    randn(mGaussian_noise,Scalar::all(mean), Scalar::all(std));

    img.convertTo(mSrc_16SC,CV_16SC3);
    addWeighted(mSrc_16SC, 1.0, mGaussian_noise, 1.0, 0.0, mSrc_16SC);
    mSrc_16SC.convertTo(noise_original,img.type());


    Mat noise = img.clone();
    missing_color = false;

    for (int k = 0; k < colors.size(); ++k){
        bool found_match = false;
        for (int i = 0; i < img.rows; ++i){
            for (int j = 0; j < img.cols; ++j){
                pixel img_pixel = {img.at<Vec3b>(i, j)[BLUE_POS], img.at<Vec3b>(i, j)[GREEN_POS],
                                   img.at<Vec3b>(i, j)[RED_POS]};
                if (check_pixel_bgr(img_pixel, colors[k])) {
                    found_match = true;
                    noise.at<Vec3b>(i, j)[BLUE_POS] = noise_original.at<Vec3b>(i, j)[BLUE_POS];
                    noise.at<Vec3b>(i, j)[GREEN_POS] = noise_original.at<Vec3b>(i, j)[GREEN_POS];
                    noise.at<Vec3b>(i, j)[RED_POS] = noise_original.at<Vec3b>(i, j)[RED_POS];
                }
            }
        }
        if (!found_match){
            missing_color = true;
            cout << "Color to replace: [" << unsigned(colors[k].b) << ", " << unsigned(colors[k].g) << ", " << unsigned(colors[k].r)<< "]  (BGR format) is not present in given image in function: \"gaussian_noise_color\""<< endl;
        }
    }
    return noise;

}

void gaussian_noise_color_all(const string & input_folder_path,  const string & output_folder_path,const vector <pixel> colors, const float & mean,const float & std){
    if (colors.empty()){
        cout << "Color vector is empty, cannot perform outline search in function \" gaussian_noise_color_all\" " << endl;
        cout << "Terminating program" << endl;
        exit(1);
    }
    if (std <0){
        cout << "Incorrect std value, mean must be >= 0 and, std entered " << std << "  ,in function \"gaussian_noise_color_all\"" << endl;
        cout << "Terminating program" << endl;
        exit(1);
    }

    if (colors.empty()){
        cout << "Color vector is empty, cannot perform outline search in function \" gaussian_noise_color_all\" " << endl;
        cout << "Terminating program" << endl;
        exit(1);
    }
    int n = 0;
    int f = 0;
    for (const auto &dirEntry : recursive_directory_iterator(input_folder_path)) {
        bool check_local = false;
        auto path = dirEntry.path();

        if (path.extension() == ".png") {

            auto dir = path.parent_path();
            auto stem = path.stem();
            auto base = dir.string() + "/" + stem.string();
            auto png = base + ".png";

            if (std::filesystem::exists(png)) {
                cv::Mat img = cv::imread(png);

                if (img.empty()) {
                    std::cout << "Could not read the image" << png << " in \"gaussian_noise_color_all\"" << std::endl;
                    continue;
                    f++;
                }
                else {
                    Mat gaussian_noise= gaussian_noise_color(img,colors,mean,std);
                    if (missing_color){
                        cout << "Image path: " << png << endl;
                    }
                    imwrite(output_folder_path+"/"+string(stem)+"_gn.png",gaussian_noise);
                    n++;
                }
            }
            else cout << "Unable to open " << png << " in \"gaussian_noise_color_all\"" << endl;
        }
    }


    printf("gaussian_noise_color_all status: %d entries successful, %d wrong\n", n, f);
}



//test functions
Mat transparency_shapes(const Mat&image, const float& alpha){
    if ((alpha<0)||(alpha>1)){
        cout << "Alpha value is not valid: " << alpha << ", must be 0 < alpha < 1, in function \"transparency_shapes\" " << endl;
        cout << "Terminating program" << endl;
        exit(EXIT_FAILURE);
    }

    Mat transparent_image_2 = image.clone();
    //cout << unsigned(colors[0].b) << " " << unsigned(colors[0].g) << " " << unsigned(colors[0].r) << endl;
    //cout << unsigned(image.at<Vec3b>(429, 513)[BLUE_POS]) << " " << unsigned(image.at<Vec3b>(429, 513)[GREEN_POS]) << " " << unsigned(image.at<Vec3b>(429, 513)[RED_POS]) << endl;

    //if ((colors[0].b == image.at<Vec3b>(429, 513)[BLUE_POS]) && (colors[0].g == image.at<Vec3b>(429, 513)[GREEN_POS]) && (colors[0].r == image.at<Vec3b>(429, 513)[RED_POS])) cout << "Match" << endl;

    //opacity*original + (1-opacity)*background


    for (int i = 0; i < image.rows; ++i) {
        for (int j = 0; j < image.cols; ++j) {
            //if (((i < image.rows/2)&&(j < image.cols/2))||((i > image.rows/2)&&(j > image.cols/2))) {
            //if (-0.75*j+600 < i) {
            //if (400 - (400-i)*(400-i) - (300-j)*(300-j) > 0){
            //if (is_inside_shape(i,j,))
            if (((i > 358)&&(i<500)&&(j > 358)&&(j<500))||(400 - (400-i)*(400-i) - (200-j)*(300-j) > 0)) {
                pixel img_pixel = {image.at<Vec3b>(i, j)[BLUE_POS], image.at<Vec3b>(i, j)[GREEN_POS],
                                   image.at<Vec3b>(i, j)[RED_POS]};

                img_pixel.b = uchar(alpha * img_pixel.b + (1 - alpha) * WHITE_BACKGROUND);
                img_pixel.g = uchar(alpha * img_pixel.g + (1 - alpha) * WHITE_BACKGROUND);
                img_pixel.r = uchar(alpha * img_pixel.r + (1 - alpha) * WHITE_BACKGROUND);
                //cout << "Global" << unsigned(img_pixel.b) << " " << unsigned(img_pixel.g) << " " << unsigned(img_pixel.r) << endl;
                transparent_image_2.at<Vec3b>(i, j)[BLUE_POS] = unsigned(img_pixel.b);
                transparent_image_2.at<Vec3b>(i, j)[GREEN_POS] = unsigned(img_pixel.g);
                transparent_image_2.at<Vec3b>(i, j)[RED_POS] = unsigned(img_pixel.r);
            }
            //}

        }
    }

    return transparent_image_2;
}

//dosen't work ? Changes the color
Mat transparency_local(const Mat& image, const vector<pixel>& colors, const float& alpha){
    if (colors.empty()){
        cout << "Color vector is empty, cannot perform outline search in function \"transparency_local\" " << endl;
        cout << "Terminating program" << endl;
        exit(EXIT_FAILURE);
    }
    if ((alpha<0)||(alpha>1)){
        cout << "Alpha value is not valid: " << alpha << ", must be 0 < alpha < 1, in function \"transparency_local\" " << endl;
        cout << "Terminating program" << endl;
        exit(EXIT_FAILURE);
    }

    Mat transparent_image = image.clone();

    for (int k = 0; k < colors.size(); ++k){
        bool found_match = false;
        for (int i = 0; i < image.rows; ++i){
            for (int j = 0; j < image.cols; ++j){
                pixel img_pixel = {image.at<Vec3b>(i, j)[BLUE_POS], image.at<Vec3b>(i, j)[GREEN_POS],
                                   image.at<Vec3b>(i, j)[RED_POS]};
                if (check_pixel_bgr(img_pixel, colors[k])) {
                    found_match = true;
                    img_pixel.b = uchar(alpha*img_pixel.b + (1-alpha)*WHITE_BACKGROUND);
                    img_pixel.g = uchar(alpha*img_pixel.g + (1-alpha)*WHITE_BACKGROUND);
                    img_pixel.r = uchar(alpha*img_pixel.r + (1-alpha)*WHITE_BACKGROUND);
                    //cout << "Local" << unsigned(img_pixel.b) << unsigned(img_pixel.b) << unsigned(img_pixel.g) << unsigned(img_pixel.r) << endl;
                    transparent_image.at<Vec3b>(i, j)[BLUE_POS] = unsigned(img_pixel.b);
                    transparent_image.at<Vec3b>(i, j)[GREEN_POS] = unsigned(img_pixel.g);
                    transparent_image.at<Vec3b>(i, j)[RED_POS] = unsigned(img_pixel.r);
                }
            }
        }
        if (!found_match){
            cout << "Color to change transparency : [" << unsigned(colors[k].b) << ", " << unsigned(colors[k].g) << ", " << unsigned(colors[k].r)<< "]  (BGR format) is not present in given image in function: \"transparency_local\""<< endl;
        }
    }
    return transparent_image;
}

Mat local_transparency(const Mat& image, const Mat& transparent_image, const vector<pixel>& colors, const float& alpha){
    if (colors.empty()){
        cout << "Color vector is empty, cannot perform outline search in function \"local_transparency\" " << endl;
        cout << "Terminating program" << endl;
        exit(EXIT_FAILURE);
    }
    if ((alpha<0)||(alpha>1)){
        cout << "Alpha value is not valid: " << alpha << ", must be 0 < alpha < 1, in function \"local_transparency\" " << endl;
        cout << "Terminating program" << endl;
        exit(EXIT_FAILURE);
    }
    vector<pixel> transparent_colors;
    vector<pixel> colors_copy  (colors);
    Mat transparent_out = image.clone();

    for (int k = 0; k < colors.size(); ++k) {
        bool found_match = false;
        for (int i = 0; i < image.rows; ++i) {
            for (int j = 0; j < image.cols; ++j) {
                pixel img_pixel = {image.at<Vec3b>(i, j)[BLUE_POS], image.at<Vec3b>(i, j)[GREEN_POS],
                                   image.at<Vec3b>(i, j)[RED_POS]};
                if (check_pixel_bgr(img_pixel, colors[k])) {
                    found_match = true;
                    pixel img_transp_pixel = {transparent_image.at<Vec3b>(i, j)[BLUE_POS], transparent_image.at<Vec3b>(i, j)[GREEN_POS],
                                              transparent_image.at<Vec3b>(i, j)[RED_POS]};
                    transparent_colors.push_back(img_transp_pixel);
                    break;
                }
            }
            if (found_match) break;
        }
        if (!found_match){
            cout << "Color to extract blended BGR : [" << unsigned(colors[k].b) << ", " << unsigned(colors[k].g) << ", " << unsigned(colors[k].r)<< "]  (BGR format) is not present in given image in function: \"local_transparency\""<< endl;
            colors_copy[k].b = unsigned(1);
            colors_copy[k].g = unsigned(0);
            colors_copy[k].r = unsigned(1);
        }
    }

    for (int k = 0; k < colors_copy.size(); ++k) {
        if ((colors_copy[k].b == 1) && (colors_copy[k].g == 0) && (colors_copy[k].r == 1)) {
            continue;
        }
        bool found_match = false;

        for (int i = 0; i < image.rows; ++i) {
            for (int j = 0; j < image.cols; ++j) {
                pixel img_pixel = {image.at<Vec3b>(i, j)[BLUE_POS], image.at<Vec3b>(i, j)[GREEN_POS],
                                   image.at<Vec3b>(i, j)[RED_POS]};
                if (check_pixel_bgr(img_pixel, colors_copy[k])) {
                    found_match = true;
                    transparent_out.at<Vec3b>(i, j)[BLUE_POS] = unsigned(transparent_colors[k].b);
                    transparent_out.at<Vec3b>(i, j)[GREEN_POS] = unsigned(transparent_colors[k].g);
                    transparent_out.at<Vec3b>(i, j)[RED_POS] = unsigned(transparent_colors[k].r);
                }
            }
        }
        if (!found_match){
            cout << "Color to replace: [" << unsigned(colors[k].b) << ", " << unsigned(colors[k].g) << ", " << unsigned(colors[k].r)<< "]  (BGR format) is not present in given image in function: \"local_transparency\""<< endl;
        }
    }
    return transparent_out;
}