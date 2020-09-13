#ifndef UTILS_H
#define UTILS_H

void del_arg(int argc, char **argv, int index);
int find_arg(int argc, char* argv[], char *arg);
int find_int_arg(int argc, char **argv, char *arg, int def);
float find_float_arg(int argc, char **argv, char *arg, float def);
char *find_char_arg(int argc, char **argv, char *arg, char *def);
int *random_index_order(int min, int max);
char *copy_string(char *s);
char *basecfg(char *cfgfile);
void find_replace(char *str, char *orig, char *rep, char *output);
void mean_cpu(float *x, int batch, int filters, int spatial, float *mean);
void variance_cpu(float *x, float *mean, int batch, int filters, int spatial, float *variance);
#endif

