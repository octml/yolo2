#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>
#include <unistd.h>
#include <float.h>
#include <limits.h>
#include <time.h>




void del_arg(int argc, char **argv, int index)
{
    int i;
    for(i = index; i < argc-1; ++i) argv[i] = argv[i+1];
    argv[i] = 0;
}

int find_arg(int argc, char* argv[], char *arg)
{
    int i;
    for(i = 0; i < argc; ++i) {
        if(!argv[i]) continue;
        if(0==strcmp(argv[i], arg)) {
            del_arg(argc, argv, i);
            return 1;
        }
    }
    return 0;
}

int find_int_arg(int argc, char **argv, char *arg, int def)
{
    int i;
    for(i = 0; i < argc-1; ++i){
        if(!argv[i]) continue;
        if(0==strcmp(argv[i], arg)){
            def = atoi(argv[i+1]);
            del_arg(argc, argv, i);
            del_arg(argc, argv, i);
            break;
        }
    }
    return def;
}

float find_float_arg(int argc, char **argv, char *arg, float def)
{
    int i;
    for(i = 0; i < argc-1; ++i){
        if(!argv[i]) continue;
        if(0==strcmp(argv[i], arg)){
            def = atof(argv[i+1]);
            del_arg(argc, argv, i);
            del_arg(argc, argv, i);
            break;
        }
    }
    return def;
}

char *find_char_arg(int argc, char **argv, char *arg, char *def)
{
    int i;
    for(i = 0; i < argc-1; ++i){
        if(!argv[i]) continue;
        if(0==strcmp(argv[i], arg)){
            def = argv[i+1];
            del_arg(argc, argv, i);
            del_arg(argc, argv, i);
            break;
        }
    }
    return def;
}

int *random_index_order(int min, int max)
{
    int *inds = calloc(max-min, sizeof(int));
    int i;
    for(i = min; i < max; ++i){
        inds[i] = i;
    }
    for(i = min; i < max-1; ++i){
        int swap = inds[i];
        int index = i + rand()%(max-i);
        inds[i] = inds[index];
        inds[index] = swap;
    }
    return inds;
}

char *copy_string(char *s)
{
    char *copy = malloc(strlen(s)+1);
    strncpy(copy, s, strlen(s)+1);
    return copy;
}
char *basecfg(char *cfgfile)
{
    char *c = cfgfile;
    char *next;
    //printf("%s\n",strchr(c, '/'));
    while((next = strchr(c, '/')))
    {
        c = next+1;
      //  printf("%s\n",c);
    }
    c = copy_string(c);
    printf("%s\n",c);
    
    next = strchr(c, '.');
    printf("%s\n",c);
    printf("next = %s\n",next);
    if (next) *next = 0;
    printf("c =%s\n",c);
    return c;
}

void find_replace(char *str, char *orig, char *rep, char *output)
{
    char buffer[4096] = {0};
    char *p;

    sprintf(buffer, "%s", str);
    //printf("buffer:%s\n",buffer);
    if(!(p = strstr(buffer, orig))){  // Is 'orig' even in 'str'?
       sprintf(output, "%s", str);
        return;
    }

     *p = '\0';

    sprintf(output, "%s%s%s", buffer, rep, p+strlen(orig));
}

void mean_cpu(float *x, int batch, int filters, int spatial, float *mean)
{
    float scale = 1./(batch * spatial);//求分母
    int i,j,k;
    for(i = 0; i < filters; ++i){
        mean[i] = 0;
        for(j = 0; j < batch; ++j){
            for(k = 0; k < spatial; ++k){
                int index = j*filters*spatial + i*spatial + k;
                mean[i] += x[index];
            }
        }
        mean[i] *= scale;
    }
}


void variance_cpu(float *x, float *mean, int batch, int filters, int spatial, float *variance)
{
    float scale = 1./(batch * spatial - 1);//注意这里的减1操作
    int i,j,k;
    for(i = 0; i < filters; ++i){
        variance[i] = 0;
        for(j = 0; j < batch; ++j){
            for(k = 0; k < spatial; ++k){
                int index = j*filters*spatial + i*spatial + k;
                variance[i] += pow((x[index] - mean[i]), 2);
            }
        }
        variance[i] *= scale;
    }
}

void reorg_cpu(float *x, int w, int h, int c, int batch, int stride, int forward, float *out)
{
    int b,i,j,k;
    int out_c = c/(stride*stride);

    for(b = 0; b < batch; ++b){
        for(k = 0; k < c; ++k){
            for(j = 0; j < h; ++j){
                for(i = 0; i < w; ++i){
                    int in_index  = i + w*(j + h*(k + c*b));
                    int c2 = k % out_c;//6%3 = 0 7%3 = 1;
                    int offset = k / out_c;
                    int w2 = i*stride + offset % stride;
                    int h2 = j*stride + offset / stride;
                    int out_index = w2 + w*stride*(h2 + h*stride*(c2 + out_c*b));
                    if(forward) out[out_index] = x[in_index];
                    else out[in_index] = x[out_index];
                }
            }
        }
    }
}
void test_mean(){
// http://blog.csdn.net/qq_17550379/article/details/78850099
    float x[] = {95,107,107,95,1,2,3,4};
    int batch = 1;
    int filters = 2;
    int spatial = 4;//2x2;//= 4;
    float mean[2]={0,0};
    mean_cpu(x, batch, filters, spatial,mean);
    printf("mean %1.2f %1.2f\n",mean[0],mean[1]);
    //101,2.5
}

void test_variance(){
// http://blog.csdn.net/qq_17550379/article/details/78850099
    float x[] = {95,107,107,95,1,2,3,4};
    int batch = 1;
    int filters = 2;
    int spatial = 4;//2x2;//= 4;
    float mean[2]={101,2.5};
    float variance[2] ={0,0};
    
    variance_cpu(x, mean, batch, filters, spatial, variance);
    printf("vari %1.2f %1.2f\n",variance[0],variance[1]);
    //48,1.667
}



float colors[6][3] = { {1,0,1}, {0,0,1},{0,1,1},{0,1,0},{1,1,0},{1,0,0} };
//深红色 255 0 255
//blue  0 0 255 (R/G/B)
//青色  0 255 255
//GREEN 0 255 0
//黄色  255 255 0
//Red   255 0 0



float get_color(int c, int x, int max)
{//这个函数是为了实现颜色的获取，使用的地方是在 draw_detections 中
    float ratio = ((float)x/max)*5;
    int i = floor(ratio); //其功能是“向下取整”
    int j = ceil(ratio);  //上取整
    ratio -= i;
    printf("i:j:ratio %d %d %1.2f\n",i,j,ratio);
    float r = (1-ratio) * colors[i][c] + ratio*colors[j][c];
    printf("color:%1.0f %1.0f\n",colors[i][c],colors[j][c]);
    //printf("%f\n", r);
    return r;
}

void test_get_color(){
    int x = 3;
    int max = 5;
    int c = 1;
    float t = get_color(c,x,max);
    printf("get color:%1.1f\n",t);
}
int main(){
    //test_mean();
    //test_variance();
    test_get_color();
    return 0;
}
