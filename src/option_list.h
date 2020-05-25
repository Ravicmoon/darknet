#ifndef OPTION_LIST_H
#define OPTION_LIST_H
#include "list.h"
#include "yolo_core.h"


typedef struct
{
  char* key;
  char* val;
  int used;
} kvp;

#ifdef __cplusplus
extern "C" {
#endif

list* ReadDataCfg(char const* filename);
int ReadOption(char* s, list* options);
void InsertOption(list* l, char* key, char* val);
char* FindOption(list* l, char* key);
char* FindOptionStr(list* l, char* key, char* def);
char* FindOptionStrQuiet(list* l, char* key, char* def);
int FindOptionInt(list* l, char* key, int def);
int FindOptionIntQuiet(list* l, char* key, int def);
float FindOptionFloat(list* l, char* key, float def);
float FindOptionFloatQuiet(list* l, char* key, float def);
void UnusedOption(list* l);

#ifdef __cplusplus
}
#endif
#endif
