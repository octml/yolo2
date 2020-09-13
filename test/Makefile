
# 可执行文件  
TARGET=test  
# C文件  
SRCS=image.c \
     blas.c \
     box.c \
	 list.c \
	 utils.c \
     test.c  
# 目标文件  变量替换函数，把所有的*.c文件替换为*.o文件
OBJS=$(SRCS:.c=.o)  
  
# 指令编译器和选项  
CC=gcc  
CFLAGS=-Wall -std=gnu99  
LDFLAGS=-lm
  
$(TARGET):$(OBJS)  
	@echo TARGET:$@  
	@echo OBJECTS:$^  
	$(CC) -o $@ $^ $(LDFLAGS) 
	./test

clean:  
	rm -rf $(TARGET) $(OBJS)  

%.o:%.c  
	$(CC) $(CFLAGS) -o $@ -c $<  

#	$@ 当前规则的目标文件
#	$< 第一个依赖文件
#   $^ 去除重复的所有依赖文件
#   http://blog.csdn.net/xukai871105/article/details/36475319