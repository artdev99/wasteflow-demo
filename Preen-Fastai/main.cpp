#include <iostream>
#include <string>
#include <vector>
#include <Tools.h>

using namespace std;
using namespace cv;

vector<string> image_paths {samples::findFile("/home/eco/CLionProjects/Mask/cmake-build-debug/original.jpg"),
                            samples::findFile("/home/eco/CLionProjects/Mask/cmake-build-debug/00014_l.png")};

int main(){
    cout << "no functions called in main" << endl;
}

//Calling examples:

//resize_all_images("/home/eco/aix/fastai/Detailed/input_test",  "/home/eco/aix/fastai/Detailed/output_test", SQUARE_SIZE);

//check_non_labeled_images("/home/eco/aix/fastai/Detailed/input_test");

//check_labeled_images("/home/eco/aix/fastai/Detailed/input_test");

//Mat img = open_image(image_paths[0]);
//vector<pixel> colors = {yellow,magenta};
//write_outlines_coord(img, colors, "/home/eco/CLionProjects/Mask/cmake-build-debug/outline_coords.txt");

//vector<pixel> colors = {red,green,magenta};
//extract_all("/home/eco/aix/fastai/Detailed/Training","/home/eco/aix/fastai/Detailed/output_modified_test",colors);

//vector<pixel> colors = {red};
//modify_all("/home/eco/aix/fastai/Windows2.0","/home/eco/aix/fastai/Windows2.0",colors,magenta);

//transparency_global_all("/home/eco/aix/fastai/Detailed/Training","/home/eco/aix/fastai/Detailed/output_modified_test",0.2);
//transparency_global_alpha_all("/home/eco/aix/fastai/Detailed/Training","/home/eco/aix/fastai/Detailed/output_modified_test",0.2);

//vector<pixel> colors = {red,green,blue};
//transparency_local_alpha_all("/home/eco/aix/fastai/Detailed/Training","/home/eco/aix/fastai/Detailed/output_modified_test", colors,0.2);

//Mat img = open_image(image_paths[0]);
//Mat img2 = open_image(image_paths[1]);
//vector <pixel> colors = {red,yellow};
//Mat original_outlines = outlines_on_original(img, img2, colors);
//imwrite("original_with_outlines.png",original_outlines);

//vector <pixel> colors = {magenta,blue};
//outlines_on_original_all ("/home/eco/aix/fastai/Detailed/Training","/home/eco/aix/fastai/Detailed/output_modified_test", colors);

//vector <pixel> colors = {magenta};
//extracted_masks_on_original_all ("/home/eco/aix/fastai/Detailed/Training","/home/eco/aix/fastai/Windows2.0", colors);

//change_names("/home/eco/aix/fastai/Windows","/home/eco/aix/fastai/Windows2.0");

//vector <pixel> colors = {magenta};
//transparency_masks_original_all ("/home/eco/aix/fastai/Detailed/Training","/home/eco/aix/fastai/Detailed/output_modified_test", colors,0.7);

//Mat image = open_image(image_paths[0]);
//Mat label = open_image(image_paths[1]);
//vector <pixel> colors = {magenta};
//Mat image_blurred= gaussian_blur (image, label, colors,23);
//mwrite("Blured_image.jpg",image_blurred);

//vector <pixel> colors = {magenta};
//gaussian_blur_all ("/home/eco/aix/fastai/Detailed/Training",  "/home/eco/aix/fastai/Detailed/Augmented_training",colors,23);

//Mat image = open_image(image_paths[0]);
//Mat noise = gaussian_noise(image, 0,50);
//imwrite("Noise_image.jpg",noise);

//gaussian_noise_all("/home/eco/aix/fastai/Detailed/Training",  "/home/eco/aix/fastai/Detailed/Augmented_training",0,50);

//gaussian_blur_global_all("/home/eco/aix/fastai/Detailed/Training",  "/home/eco/aix/fastai/Detailed/Augmented_training",23);

//gaussian_noise_all_2("/home/eco/aix/fastai/Detailed/dev/Training","/home/eco/aix/fastai/Detailed/dev/Training",0,50);

//Mat img = open_image("/home/eco/aix/fastai/Detailed/dev/Training/Acura1_gn.png");
// show_image(img);
//vector<pixel> colors = {magenta};
//Mat extract_img = extract(img,colors);
//show_image(extract_img);

//gaussian_blur_global_all_2("/home/eco/aix/fastai/Detailed/dev/Training","/home/eco/aix/fastai/Detailed/dev/Blured",23);
//gaussian_noise_all_2("/home/eco/aix/fastai/Detailed/dev/Training","/home/eco/aix/fastai/Detailed/dev/Noise",0,50);

//Mat img = open_image("/home/eco/aix/fastai/Detailed/dev/Training/Acura1_gb.png");
//show_image(img);
//vector<pixel> colors = {magenta};
//Mat extract_img = extract(img,colors);
//show_image(extract_img);

/*
Mat img = imread("/home/eco/aix/fastai/Output_Test/Acura1.png", IMREAD_GRAYSCALE  );
for (int i  = 0; i < img.rows; ++i){
    for (int j = 0; j < img.cols; ++j){
        if (img.at<uchar>(i,j) == 2){
            //cout <<unsigned(img.at<uchar>(i,j)) << endl;
            cout << i << " " << j << endl;
        }
    }
}*/

//cout << img << endl;
/*
Mat img = imread("/home/eco/aix/fastai/Output_Test/00001.png");
vector <pixel> colors = {blue};
Mat extract_img = extract(img,colors);
show_image(extract_img);*/

//gaussian_noise_all_2 ("/home/eco/aix/fastai/Detailed/dev/Augmented_Windows_395",  "/home/eco/aix/fastai/Detailed/dev/Augmented_Windows_395",0,50);
//gaussian_blur_global_all_2("/home/eco/aix/fastai/Detailed/dev/backup_training_sets/Augmented_Windows","/home/eco/aix/fastai/Detailed/dev/Augmented_Windows_2",15);

//Mat img = open_image("/home/eco/Desktop/Tesla_Model_X.PNG");
//square_and_write("/home/eco/Desktop/Tesla_Model_X_RESIZED.PNG",img,SQUARE_SIZE);

//gaussian_noise_all_2("/home/eco/aix/fastai/Detailed/dev/Set_Training_200","/home/eco/aix/fastai/Detailed/dev/Set_Aug_Training_800",0,50);
//gaussian_blur_global_all_2("/home/eco/aix/fastai/Detailed/dev/Set_Aug_Training_800","/home/eco/aix/fastai/Detailed/dev/Set_Aug_Training_800",15);

//Mat img = open_image("/home/eco/Desktop/00014_l.png");
//vector <pixel> colors = {magenta};
//Mat noise_col = gaussian_noise_color(img,colors,0,50);
//show_image(noise_col);

//vector<pixel> colors = {magenta};
//gaussian_noise_color_all("/home/eco/aix/fastai/Detailed/dev/Set_Training_200","/home/eco/aix/fastai/Detailed/dev/z_output_noise_local",colors,0,50);

/*
    Mat img = open_image(image_paths[0]);
    //show_image(img);
    imwrite("original.png",img);

    vector<pixel> extract_colors  = {dark_red};
    Mat image_extract = extract(img,extract_colors);
    //show_image(image_extract);

    //pixel fake_dark_red = hex_to_bgr("ffffff");
    //cout << unsigned(fake_dark_red.r) << " " << unsigned(fake_dark_red.g) << " " << unsigned(fake_dark_red.b) << endl;
    vector<pixel> modify_colors  = {dark_red};
    Mat image_modified = modify(img,modify_colors, blue);
    //show_image(image_modified);
    imwrite("Ferrari5.jpg",image_modified);

    //image_extract = extract(image_modified,extract_colors);
    //show_image(image_extract);
    //imwrite("extracted_ferra.png",image_extract);

    //Mat image_outlines = find_outline(image_modified,extract_colors);
    //show_image(image_outlines);
    //imwrite("outlines.png",image_outlines);

    //Mat image_transparent_global = transparency_global(image_modified,0.3);
    //show_image(image_transparent_global);
    //imwrite("Alpha_Bending.jpg",image_transparent_global);

    //Mat image_transparent_global_alpha = transparency_global_alpha(image_modified,0.3);
    //imwrite("Alpha_Channel.png",image_transparent_global_alpha);

    //Mat image_transparent_local = transparency_local(image_modified,extract_colors,0.3);
    //Mat image_transperent_local_2 = tricky_local_transparency(image_modified,image_transparent_global,extract_colors,0.5);
    //show_image(image_transperent_local);
    //imwrite("Transparency_local.jpg",image_transparent_local);

    //Mat image_transparent_local_3 = transparency_local_outline(image_modified,0.3);
    //imwrite("Transparency_local_test.png",image_transparent_local_3);

    //Mat image_transparent_local_alpha = transparency_local_alpha(image_modified,extract_colors,0.7);
    //imwrite("Transparency_local_alpha.png",image_transparent_local_alpha);

    //Mat show = transparency_local_outline(img,0.3);
    //show_image(show);

    //square_and_write("/home/eco/Desktop/Resized.png",img);

*/