
#include "utils.h"
#include "box.h"
#include "blas.h"
#include "image.h"
#include <stdio.h>

void test_string(void)
{
    //test_box();
    //test_convolutional_layer();
   // int gpu_index;
   // printf("%d,%s\n",argc,*argv);
    //test_resize("data/bad.jpg");
   // gpu_index = find_int_arg(argc, argv, "-i", 0);
   // printf("%d\n",gpu_index);
    //if(argc < 2){
    //    fprintf(stderr, "usage: %s <function>\n", argv[0]);
    //    return 0;
    //}
    /*
    int *test;
    int i;
    test = random_index_order(1, 6);

     for(i = 1; i < 6; ++i){
        printf("%d\n",test[i]);
     };

     */
    /*char * test;
    test = basecfg("a/c/b.cfg");
    printf("%s\n",test);
*/
 //   char output[20];
 //   find_replace("i am a patrick!", "patrick", "liu", output);
 //   printf("%s\n",output);
    char str[] = "I am a patrick";
	char *key;
	char *val = 0;
    printf("str:%s\n",str);
    str[4] = '\0';
	
    val = str+5;
    key =str;
	
    printf("key:%s,valu:%s\n",key,val);
}

void test_image(void){
    //image test = make_random_image(640,480,3);
    //save_image(test,"./out/test_image");
   //image test = load_image("./out/computer.png", 640, 480, 3);
   // save_image(test,"./out/test_image");
   // print_image(test);   
    //image coll_image[2];
    //coll_image[0] = load_image("./out/computer.png", 640, 480, 3);
    //coll_image[1] = load_image("./out/coll_image.png", 640, 480, 3);
    //image test_image1 = collapse_images_vert(coll_image,2);
    //save_image(test_image1,"./out/collapse_image");
   
   // image dest = load_image("./out/coll_image.png", 640, 480, 3);
   // composite_image(coll_image[0],dest,100,100);
   // save_image(dest,"./out/composite_image");

    image test = load_image("./out/computer.png", 640, 480, 3);
    image im_border = border_image(test,400);
    save_image(im_border,"./out/border_image");
    printf("save png file to out folder\n");
}
int main(){
    //test_string();
    test_image();

    return 0;
}