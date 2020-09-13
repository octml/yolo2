#include <stdlib.h>
#include <stdio.h>
#include <math.h>
void gemm_cpu(int TA, int TB, int M, int N, int K, float ALPHA, 
        float *A, int lda, 
        float *B, int ldb,
        float BETA,
        float *C, int ldc);
/*
** 该函数只是调用了gemm_cpu()函数，并且将参数原封不动的传给gemm_cpu()
*/
void gemm(int TA, int TB, int M, int N, int K, float ALPHA, 
        float *A, int lda, 
        float *B, int ldb,
        float BETA,
        float *C, int ldc)
{
    gemm_cpu( TA,  TB,  M, N, K, ALPHA,A,lda, B, ldb,BETA,C,ldc);
}

/*
**  功能：被gemm_cpu()函数调用，实际完成C = ALPHA * A * B + C 矩阵计算，
**       输出的C也是按行存储（所有行并成一行）
**  输入： A,B,C   输入矩阵（一维数组格式）
**        ALPHA   系数
**        BETA    系数
**        M       A,C的行数（不做转置）或者A'的行数（做转置），此处A未转置，故为A的行数
**        N       B,C的列数（不做转置）或者B'的列数（做转置），此处B未转置，故为B的列数
**        K       A的列数（不做转置）或者A'的列数（做转置），B的行数（不做转置）或者B'的行数（做转置），此处A,B均未转置，故为A的列数、B的行数
**        lda     A的列数（不做转置）或者A'的行数（做转置），此处A未转置，故为A的列数
**        ldb     B的列数（不做转置）或者B'的行数（做转置），此处B未转置，故为B的列数
**        ldc     C的列数
**  说明1：此函数是用C实现矩阵乘法运算，这部分代码应该是模仿的Caffe中的math_functions.cpp的代码
**       参考博客：http://www.voidcn.com/blog/thy_2014/article/p-6149690.html
**       更为详细的注释参见：gemm_cpu()函数的注释
**  说明2：此函数在gemm_cpu()函数中调用，是其中四种情况之一，A,B都不进行转置
**       函数名称gemm_nn()中的两个nn分别表示not transpose， not transpose
**    A[M,K] *B[K,N] = C[M,N]
**   [1,2,3,4,5,6,7,8,9]*[0,10,20,30,
**                        1,11,21,31,
**                        2,12,22,32,
**                        3,13,23,33,
**                        4,14,24,34,
**                        5,15,25,35,
**                        6,16,26,36,
**                        7,17,27,37,
**                        8,18,28,38] =[x,x,x,x]
**   a use 9 * 1 so lda = 1 
*/
void gemm_nn(int M, int N, int K, float ALPHA, 
        float *A, int lda, 
        float *B, int ldb,
        float *C, int ldc)
{
    int i,j,k;
    // 大循环：遍历A的每一行，i表示A的第i行，也是C的第i行
    for(i = 0; i < M; ++i){
        // 中循环：遍历每一行的所有列，k表示A的第k列，同时表示B的第k行
        for(k = 0; k < K; ++k){
            // 先计算ALPHA * A（A中每个元素乘以ALPHA）
            register float A_PART = (float)(ALPHA*A[i*lda+k]);
            //printf("nn-A:%1.0f\n",A[i*lda+k]);
            // 内循环：遍历B中所有列，每次大循环完毕，将计算得到A*B一行的结果
            // j是B的第j列，也是C的第j列
            for(j = 0; j < N; ++j){
                // A中的第i行k列与B中的k行i列对应相乘，因为一个大循环要计算A*B整行之结果，
                // 因此，这里用了一个内循环，并没有直接乘以B[k*ldb+i]
                // 每个内循环完毕，将计算A*B整行的部分结果（A中第i行k列与B所有列第k行所有元素相乘的结果）
                C[i*ldc+j] += A_PART*B[k*ldb+j];
               // printf("nn-A:%1.0f\n",A_PART);
               // printf("nn-B:%1.0f\n",B[k*ldb+j]);
               // printf("nn-C0:%1.0f\n",A_PART*B[k*ldb+j]);
               // printf("nn-C:%1.0f\n",C[i*ldc+j]);
            }
            
        }
    }
}

/*
**  功能：被gemm_cpu()函数调用，实际完成C = ALPHA * A * B' + C矩阵计算，
**       输出的C也是按行存储（所有行并成一行）
**  输入： A,B,C   输入矩阵（一维数组格式）
**        ALPHA   系数
**        BETA    系数
**        M       A,C的行数（不做转置）或者A'的行数（做转置），此处A未转置，故为A的行数
**        N       B,C的列数（不做转置）或者B'的列数（做转置），此处B转置，故为B’的列数
**        K       A的列数（不做转置）或者A'的列数（做转置），B的行数（不做转置）或者B'的行数（做转置），此处A不转置，B转置，故为A的列数、B'的行数
**        lda     A的列数（不做转置）或者A'的行数（做转置），此处A未转置，故为A的列数
**        ldb     B的列数（不做转置）或者B'的行数（做转置），此处B未转置，故为B'的行数
**        ldc     C的列数
**  说明：此函数是用C实现矩阵乘法运算，这部分代码应该是模仿的Caffe中的math_functions.cpp的代码
**       参考博客：http://www.voidcn.com/blog/thy_2014/article/p-6149690.html
**       更为详细的注释参见：gemm_cpu()函数的注释
**  说明2：此函数在gemm_cpu()函数中调用，是其中四种情况之一，A不进行转置,B转置
**       函数名称gemm_nt()中的nt分别表示not transpose， transpose
*/
void gemm_nt(int M, int N, int K, float ALPHA, 
        float *A, int lda, 
        float *B, int ldb,
        float *C, int ldc)
{
    int i,j,k;
    // 大循环：遍历A的每一行，i表示A的第i行，也是C的第i行
    for(i = 0; i < M; ++i){
        // 
        for(j = 0; j < N; ++j){
            register float sum = 0;
            // 内循环：每次内循环结束，将计算A中第i行与B中第j列相乘的结果，
            // 也就是得到C[i][j]，因为C也一维化了，且按行存储，所以得到C[i*lda+j]
            // k表示A的第几列，也表示
            for(k = 0; k < K; ++k){
                sum += ALPHA*A[i*lda+k]*B[j*ldb + k];
            }
            C[i*ldc+j] += sum;
        }
    }
}

/*
**  功能：矩阵计算，实际完成C = ALPHA * A' * B + BETA * C矩阵计算
**  输入： A,B,C   输入矩阵（一维数组格式）
**        ALPHA   系数
**        BETA    系数
**        M       A,C的行数（不做转置）或者A'的行数（做转置），此处A转置，故为A'的行数
**        N       B,C的列数（不做转置）或者B'的列数（做转置），此处B未转置，故为B的列数
**        K       A的列数（不做转置）或者A'的列数（做转置），B的行数（不做转置）或者B'的行数（做转置），此处A转置，B不转置，故为A'的列数、B的行数
**        lda     A的列数（不做转置）或者A'的行数（做转置），此处A转置，故为A'的行数  
**        ldb     B的列数（不做转置）或者B'的行数（做转置），此处B未转置，故为B的列数
**        ldc     C的列数
**  说明：此函数是用C实现矩阵乘法运算，这部分代码应该是模仿的Caffe中的math_functions.cpp的代码
**       参考博客：http://www.voidcn.com/blog/thy_2014/article/p-6149690.html
**       更为详细的注释参见：gemm_cpu()函数的注释
**  说明2：此函数在gemm_cpu()函数中调用，是其中四种情况之一，A进行转置,B不转置
**       函数名称gemm_tn()中的tn分别表示transpose， not transpose
*/
void gemm_tn(int M, int N, int K, float ALPHA, 
        float *A, int lda, 
        float *B, int ldb,
        float *C, int ldc)
{
    int i,j,k;
    for(i = 0; i < M; ++i){
        for(k = 0; k < K; ++k){
            register float A_PART = ALPHA*A[k*lda+i];
            for(j = 0; j < N; ++j){
                C[i*ldc+j] += A_PART*B[k*ldb+j];
            }
        }
    }
}

/*
**  功能：矩阵计算，实际完成C = ALPHA * A' * B' + BETA * C矩阵计算
**  输入： A,B,C   输入矩阵（一维数组格式）
**        ALPHA   系数
**        BETA    系数
**        M       A,C的行数（不做转置）或者A'的行数（做转置），此处A转置，故为A'的行数
**        N       B,C的列数（不做转置）或者B'的列数（做转置），此处B转置，故为B'的列数
**        K       A'的列数，B'的行数
**        lda     A的列数（不做转置）或者A'的行数（做转置），此处A转置，故为A'的行数  
**        ldb     B的列数（不做转置）或者B'的行数（做转置），此处B转置，故为B'的行数
**        ldc     C的列数
**  说明：此函数是用C实现矩阵乘法运算，这部分代码应该是模仿的Caffe中的math_functions.cpp的代码
**       参考博客：http://www.voidcn.com/blog/thy_2014/article/p-6149690.html
**       更为详细的注释参见：gemm_cpu()函数的注释
**  说明2：此函数在gemm_cpu()函数中调用，是其中四种情况之一，A,B都进行转置
**       函数名称gemm_tt()中的tt分别表示transpose， transpose
*/
void gemm_tt(int M, int N, int K, float ALPHA, 
        float *A, int lda, 
        float *B, int ldb,
        float *C, int ldc)
{
    int i,j,k;
    for(i = 0; i < M; ++i){
        for(j = 0; j < N; ++j){
            register float sum = 0;
            for(k = 0; k < K; ++k){
                sum += ALPHA*A[i+k*lda]*B[k+j*ldb];
            }
            C[i*ldc+j] += sum;
        }
    }
}

/*
**  功能：矩阵计算，完成C = ALPHA * A * B + BETA * C矩阵计算，最后的输出为C
**  输入： 
**        TA,TB   是否需要对A,B做转置操作，是为1,否为0（要不要转置取决于A,B之间维度是否匹配，比如A:3*2,B:4*2，则需要对B转置，才满足矩阵乘法维度匹配规则）
**        M       A,C的行数（若A需要转置，则此处给出转置后的A即A'的行数，而不是转置前的）
**        N       B,C的列数（若B需要转置，则此处给出转置后的B即B'的列数，而不是转置前的）
**        K       A的列数，B的行数（同样，若A与B中的二者或者其中一个需要转置，则不管怎样，转置后的A，B必须行列能够匹配，符合矩阵乘法规则，K也是转置后的值，不是转置前的）
**        A,B,C   输入矩阵（一维数组格式）
**        ALPHA   系数
**        BETA    系数
**        lda     A的列数（不做转置）或者行数（做转置，且给的是转置后A即A'的行数）
**        ldb     B的列数（不做转置）或者行数（做转置，且给的是转置后B即B'的行数）
**        ldc     C的列数
**  说明：如果TA = 0, TB = 0，那么计算的是C = ALPHA * A * B + BETA * C,此时M是A,C的行数，N是B,C的列数，K是A的列数、B的行数，lda是A的列数，ldb是B的列数；
**       如果TA = 1, TB = 0，那么计算的是C = ALPHA * A' * B + BETA * C,此时M是A’,C的行数，N是B,C的列数，K是A'的列数、B的行数，lda是A'的行数，ldb是B的列数；
**       如果TA = 0, TB = 1，那么计算的是C = ALPHA * A * B' + BETA * C,此时M是A,C的行数，N是B',C的列数，K是A的列数、B'的行数，lda是A的列数，ldb是B'的行数；
**       如果TA = 1, TB = 1，那么计算的是C = ALPHA * A' * B' + BETA * C,此时M是A’,C的行数，N是B',C的列数，K是A'的列数、B'的行数，lda是A'的行数，ldb是B'的行数；
**       总之，参与计算的矩阵必须满足矩阵行列匹配规则。比如A为2*3，B为3*2，C为2*2，那么就是第一种情况；而如果A为3*2，B为3*2，C为2*2,
**       那么就是第二种情况；如果A为2*3，B为2*3，C为2*2,对应第三种情况；如果A为2*3，B为2*3，C为2*2,对应第四种情况。
**  链接：此函数是用C实现矩阵乘法运算，这部分代码应该是模仿的Caffe中的math_functions.cpp的代码
**       参考博客：http://www.voidcn.com/blog/thy_2014/article/p-6149690.html
**  举例说明： 这个函数比较难以理解的地方在于A,B有没有转置这个问题上。首先要清楚，虽然这里A,B,C都是矩阵，但其实都是用一维数组按行保存的，
**           举个例子，假设： A = [1, 2, 3, 2, 2, 1], B = [2, 0, 1, 1, 2, 1], C = [3, 0, 1, 2] （这些输入是打死不变的，
**           都是一维数组格式），且C为2*2的矩阵，即C = [3, 0; 1, 2]，那么要进行C = ALPHA * A * B + BETA * C的计算，
**           必须满足矩阵乘法行列匹配规则，则参与运算的第一个矩阵只能为2*3，第二个只能为3*2，因为A,B的元素个数已经固定为6个。
**           下面分别说明gemm_nn(),gemm_tn(),gemm_nt,gemm_tt()四个函数对该例子的计算。
**           诚如上所述，不管A, B有没有转置，反正最后参与计算的两个矩阵必须前者为2*3,后者为3*2。如果使用gemm_nn()，A,B都没有转置，
**           那么就要求没有转置的A,B分别为2*3,3*2矩阵，则 A = [ 1, 2, 3; 2, 2, 1], B = [2, 0; 1, 1; 2, 1], 
**           调用gemm_nn(2, 2, 3, 1, A, 3, B, 2, C, 2)计算得到 C = [13, 5; 9, 5]（其中ALPHA = BETA = 1，下同）；

**           如果要用gemm_tn()函数，即A需要进行转置之后才能计算，也即转置之后的维度为2*3,而转置之前的维度为3*2，B没有转置，
**           本身就是3*2的矩阵，这样，A = [ 1, 2; 3, 2; 2, 1], A' = [1, 3, 2; 2, 2, 1], B = [2, 0; 1, 1; 2, 1]，
**           gemm_tn(2, 2, 3, 1, A, 2, B, 2, C, 2)函数实际计算的是A'*B+C的值，注意此时的A与gemm_nn()中的A有什么不同，
**           输入的一维数组还是[1, 2, 3, 2, 2, 1]，如前所述，A是按行保存的，因为此时的A本身是一个3*2的矩阵，按照按行保存规则，
**           就是A = [ 1, 2; 3, 2; 2, 1]，调用gemm_tn()的时候，M, N, K分别为2, 2, 3,都是最终参与计算的矩阵的行列数，
**           因为此处真正参与计算的是A'与B，所以M为A'的行数，即为2,N为B的列数，即为2,K为A'与B的列数，即为3，而此时lda=2，
**           是因为A进行了转置，因此输入的是A'的行数，而不是列数3,ldb=2，为B的列数，最终计算得到C=[12, 5; 9, 5]。
**           对于gemm_nt()与gemm_tt()，与上分析一样，不再赘述了。此部分注释进行了测试，对应测试文件darknet_test_gemm.c。
**  强调： 这一系列的gemm()函数，都带有叠加效果，也即最终的值是保存在C中，但这种保存并不是擦除式的保存，而是叠加式的保存，也就是说，
**        如果进入gemm()函数之前，如果C的元素已经有值了，那么这些值不会被擦除掉，而是会将其叠加，
**        其实看式子就可以看出来：此函数完成的是C = ALPHA * A * B + BETA * C矩阵运算。
**          
*/
void gemm_cpu(int TA, int TB, int M, int N, int K, float ALPHA, 
        float *A, int lda, 
        float *B, int ldb,
        float BETA,
        float *C, int ldc)
{
    //printf("cpu: %d %d %d %d %d %f %d %d %f %d\n",TA, TB, M, N, K, ALPHA, lda, ldb, BETA, ldc);
    int i, j;
    // 先把BETA * C计算完了，并将结果存在C中，得到的C将为M行，N列（按行存储在一维数组C中）
    for(i = 0; i < M; ++i){
        for(j = 0; j < N; ++j){
            C[i*ldc + j] *= BETA;
        }
    }
    // 根据需要，调用下面四种函数之一
    if(!TA && !TB)
        gemm_nn(M, N, K, ALPHA,A,lda, B, ldb,C,ldc);
    else if(TA && !TB)
        gemm_tn(M, N, K, ALPHA,A,lda, B, ldb,C,ldc);
    else if(!TA && TB)
        gemm_nt(M, N, K, ALPHA,A,lda, B, ldb,C,ldc);
    else
        gemm_tt(M, N, K, ALPHA,A,lda, B, ldb,C,ldc);
}

float im2col_get_pixel(float *im, int height, int width, int channels,
                        int row, int col, int channel, int pad)
{
   // printf("H:W:C %d %d %d --r:c:c:p %d %d %d %d\n",height,width,channels, row,col,channel, pad);
    row -= pad; // 减去补0长度，获取元素真实的行列数
    col -= pad;
    //printf("im2col\n");
    //printf("%d",row);
    // 如果行列数小于0,则返回0（刚好是补0的效果）
    if (row < 0 || col < 0 ||
        row >= height || col >= width) return 0;
    // im存储多通道二维图像的数据的格式为：各通道所有行并成一行，再多通道依次并成一行，
    // 因此width*height*channel首先移位到所在通道的起点位置，加上width*row移位到所在指定通道所在行，再加上col移位到所在列    
    int position = col + width*(row + height*channel);
    //printf("%d:%1.0f\n",position,im[position]);
    return im[position];
    
    
}
void im2col_cpu(float* data_im,
     int channels,  int height,  int width,
     int ksize,  int stride, int pad, float *data_col) 
{
    int c,h,w;
    // 计算该层神经网络的输出图像尺寸（其实没有必要再次计算的，因为在构建卷积层时，make_convolutional_layer()函数
    // 已经调用convolutional_out_width()，convolutional_out_height()函数求取了这两个参数，
    // 此处直接使用l.out_h,l.out_w即可，函数参数只要传入该层网络指针就可了，没必要弄这么多参数）
    int height_col = (height + 2*pad - ksize) / stride + 1;
    int width_col = (width + 2*pad - ksize) / stride + 1;
     
    // 卷积核大小：ksize*ksize是一个卷积核的大小，之所以乘以通道数channels，是因为输入图像有多通道，每个卷积核在做卷积时，
    // 是同时对同一位置处多通道的图像进行卷积运算，这里为了实现这一目的，将三通道上的卷积核并在一起以便进行计算，实际就是同一个卷积核的复制，
    // 比如对于3通道图像，卷积核尺寸为3*3，该卷积核将同时作用于三通道图像上，这样并起来就得到含有27个元素的卷积核
    // 能不能使得作用在不同通道上的卷积核有不同参数呢？不知道有没有这样的做法？可以思考下，当然这样做肯定会是参数剧增！！
    
    int channels_col = channels * ksize * ksize;
  //  printf("channel_col = %d height_col = %d width_col = %d\n",channels_col,height_col,width_col);
    printf("conv size: %d -- H:W %d %d\n",height_col*width_col ,height_col,width_col);

    // ******这三层循环之间的逻辑关系，决定了输入图像重排后的格式，更为详细/形象的说明可参考博客*******
    // 外循环次数为一个卷积核的尺寸数，循环次数即为最终得到的data_col的总行数
    for (c = 0; c < channels_col; ++c) {
        // 列偏移，卷积核是一个二维矩阵，并按行存储在一维数组中，利用求余运算获取对应在卷积核中的列数，比如对于
        // 3*3的卷积核（3通道），当c=0时，显然在第一列，当c=5时，显然在第2列，当c=9时，在第二通道上的卷积核的第一列，
        // 当c=26时，在第三列（第三通道上）
        int w_offset = c % ksize;
        // 行偏移，卷积核是一个二维的矩阵，且是按行（卷积核所有行并成一行）存储在一维数组中的，
        // 比如对于3*3的卷积核，处理3通道的图像，那么一个卷积核具有27个元素，每9个元素对应一个通道上的卷积核（互为一样），
        // 每当c为3的倍数，就意味着卷积核换了一行，h_offset取值为0,1,2，对应3*3卷积核中的第1, 2, 3行
        int h_offset = (c / ksize) % ksize;
        // 通道偏移，channels_col是多通道的卷积核并在一起的，比如对于3通道，3*3卷积核，每过9个元素就要换一通道数，
        // 当c=0~8时，c_im=0;c=9~17时，c_im=1;c=18~26时，c_im=2
        int c_im = c / ksize / ksize;

        // 中循环次数等于该层输出图像行数height_col，说明data_col中的每一行存储了一张特征图，这张特征图又是按行存储在data_col中的某行中
        for (h = 0; h < height_col; ++h) {
            // 内循环等于该层输出图像列数width_col，说明最终得到的data_col总有channels_col行，height_col*width_col列
            for (w = 0; w < width_col; ++w) {
                // 由上面可知，对于3*3的卷积核，h_offset取值为0,1,2,当h_offset=0时，会提取出所有与卷积核第一行元素进行运算的像素，
                // 依次类推；加上h*stride是对卷积核进行行移位操作，比如卷积核从图像(0,0)位置开始做卷积，那么最先开始涉及(0,0)~(3,3)
                // 之间的像素值，若stride=2，那么卷积核进行一次行移位时，下一行的卷积操作是从元素(2,0)（2为图像行号，0为列号）开始
                int im_row = h_offset + h * stride;
                // 对于3*3的卷积核，w_offset取值也为0,1,2，当w_offset取1时，会提取出所有与卷积核中第2列元素进行运算的像素，
                // 实际在做卷积操作时，卷积核对图像逐行扫描做卷积，加上w*stride就是为了做列移位，
                // 比如前一次卷积其实像素元素为(0,0)，若stride=2,那么下次卷积元素起始像素位置为(0,2)（0为行号，2为列号）
                int im_col = w_offset + w * stride;

                // col_index为重排后图像中的像素索引，等于c * height_col * width_col + h * width_col +w（还是按行存储，所有通道再并成一行），
                // 对应第c通道，h行，w列的元素
                int col_index = (c * height_col + h) * width_col + w;
                
                // im2col_get_pixel函数获取输入图像data_im中第c_im通道，im_row,im_col的像素值并赋值给重排后的图像，
                // height和width为输入图像data_im的真实高、宽，pad为四周补0的长度（注意im_row,im_col是补0之后的行列号，
                // 不是真实输入图像中的行列号，因此需要减去pad获取真实的行列号）
                //printf("H:W:C %d %d %d --r:c:c:p %d %d %d %d\n",height,width,channels, im_row, im_col, c_im, pad);
                data_col[col_index] = im2col_get_pixel(data_im, height, width, channels,im_row, im_col, c_im, pad);
                
               // float re = im2col_get_pixel(data_im, height, width, channels,im_row, im_col, c_im, pad);
                //printf("col_index:d %d %1.0f\n",col_index,data_col[col_index]);
               // data_col[col_index] = re;
               //printf("%1.0f ",data_col[col_index]);
            }
            //printf("\n");
        }
        //printf("\n");
    }
}

void print_array(float *array,int height,int width){
    int r,c;
    for(r = 0;r<height;r++){
       for(c=0;c<width;c++){
            printf("%1.0f ",array[r*width+c]);
       }
        printf("\n");
    }
}



//conv demo as  http://m.blog.csdn.net/mrhiuser/article/details/52672824
int main1(){


    printf("gemm test!\n");
   
    /*
    float data_im[] = {1,2,6,4,8,\
                 4,2,7,5,2,\
                 1,5,9,0,2,\
                 3,5,8,5,1,\
                 3,1,7,4,9};
    int channels = 1;
    int height = 5;
    int width = 5;
    int ksize = 3;
    int stride= 2;
    int pad = 0;
    float data_col[9];

    */
    /*
    float data_im[] = {0,1,2,3,\
                 4,5,6,7,\
                 8,9,10,11,\
                 12,13,14,15};
    int channels = 1;
    int height = 4;
    int width = 4;
    int ksize = 3;
    int stride= 1;
    int pad = 0;
    float data_col[27];
    */
    float weight_in[] = {1,1,1,\
                 1,1,1,\
                 1,1,1};
    int channels = 1;
    int height = 3;
    int width = 3;
    int ksize = 3;
    int stride= 1;
    int pad = 0;
    float a[9];
    //printf("%1.0f\n",data_im[0]);
   // print_array(data_im,height,width);
    im2col_cpu(weight_in,channels, height, width,ksize,stride, pad,a);
    print_array(a,9,1);

    float data_im[] = {0,1,2,3,\
                 4,5,6,7,\
                 8,9,10,11,\
                 12,13,14,15};
    channels = 1;
    height = 4;
    width = 4;
    ksize = 3;
    stride= 1;
    pad = 0;
    float b[36];
    im2col_cpu(data_im,channels, height, width,ksize,stride, pad,b);
    print_array(b,9,4);
    //printf("%1.0f 1.0f 1.0f 1.0f\n",data_col[0],data_col[1],data_col[2],data_col[3]);
     int M = 1; //m是卷积核的个数
    int N = 2*2; //k是每个卷积核的参数数量（l.size是卷积核的大小）
    //int l.out_h = (height + 2*pad - ksize) / stride + 1;
    //int l.out_w = (width + 2*pad - ksize) / stride + 1;
    // N = height_col * width_col
    int K = 3*3*1;//n是每个输出feature map的像素个数

    /*
    int m = l.n/l.groups; //kernel number
    int k = l.size*l.size*l.c/l.groups;
    int n = l.out_w*l.out_h;
    */
    float ALPHA = 1;
    int lda = 1;
    int ldb = 4;
    
    float conv[4]={0,0,0,0};
    int ldc = 2;
    //float a1[] = {1,1,1,1,1,1,1,1,1};
    gemm_nn(M, N, K, ALPHA, a, lda, b, ldb,conv,ldc);//[1,9]*[9,4] =[1,4]
    printf("cov result:conv(a,b)\n");
    print_array(conv,1,4);
    
   // im2col_cpu(data_im,channels, height, width,ksize,stride, pad,b);
   // print_array(b,9,4);
    return 0;
}


void test_conv1(){
    printf("demo: http://blog.csdn.net/x_r_su/article/details/53046977\n");
    float weight_in[] = {1,0,1,\
                 0,1,0,\
                 1,0,1};
     float weight[9];
    //printf("%1.0f\n",data_im[0]);
   // print_array(data_im,height,width);
    im2col_cpu(weight_in,1,3,3,3,1,0,weight);
    print_array(weight,1,9);

    float data_im[] = {1,1,1,0,0,\
                       0,1,1,1,0,\
                       0,0,1,1,1,\
                       0,0,1,1,0,\
                       0,1,1,0,0};
    int channels = 1;
    int height = 5;
    int width = 5;
    int ksize = 3;
    int stride= 1;
    int pad = 0;

    float im[81]; //((5-3+1)*(5-3+1))*9
    im2col_cpu(data_im,channels, height, width,ksize,stride, pad,im);
    print_array(im,9,9);

    float ALPHA = 1;
    int lda = 9;
    int ldb = 9;
    
    float conv[9]={0,0,0,0,0,0,0,0,0};
    int ldc = 9;
    //float a1[] = {1,1,1,1,1,1,1,1,1};
    int M = 1;
    int N = 9;
    int K = 9;
    gemm_nn(M, N, K, ALPHA, weight, lda, im, ldb,conv,ldc);//[1,9]*[9,9] =[1,9]=>3*3
    printf("cov result:conv(a,b)\n");
    print_array(conv,3,3);
    //result = 4,3,4,2,4,3,2,3,4
}


void test_local(){
    printf("local layer\n");
    float weight_in[] = {1,0,1,\
                 0,1,0,\
                 1,0,1};
     float weight[9];
    //printf("%1.0f\n",data_im[0]);
   // print_array(data_im,height,width);
    im2col_cpu(weight_in,1,3,3,3,1,0,weight);
    print_array(weight,1,9);

    float data_im[] = {1,1,1,0,0,\
                       0,1,1,1,0,\
                       0,0,1,1,1,\
                       0,0,1,1,0,\
                       0,1,1,0,0};
    int channels = 1;
    int height = 5;
    int width = 5;
    int ksize = 3;
    int stride= 1;
    int pad = 0;

    float im[81]; //((5-3+1)*(5-3+1))*9
    im2col_cpu(data_im,channels, height, width,ksize,stride, pad,im);
    print_array(im,9,9);

    float ALPHA = 1;
    int lda = 9;
    int ldb = 9;
    
    float conv[9]={0,0,0,0,0,0,0,0,0};
    int ldc = 9;
    //float a1[] = {1,1,1,1,1,1,1,1,1};
    int M = 1;
    int N = 1;
    int K = 9;
    int j = 1; //increase every cycle as weight's variable
    gemm_nn(M, N, K, ALPHA, weight, lda, im+j, ldb,conv+j,ldc);//[1,9]*[9,9] =[1,9]=>3*3
    printf("cov result:conv(a,b)\n");
    print_array(conv,3,3);
    //result = 4,3,4,2,4,3,2,3,4
}

int main(){
    test_conv1();
    //test_local();
    return 0;
}