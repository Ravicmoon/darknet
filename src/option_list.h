#pragma once
#include <string>

#include "libapi.h"
#include "list.h"

class LIB_API Metadata
{
 public:
  Metadata(char const* filename);
  ~Metadata();

  int NumClasses() const;
  std::string NameAt(int idx) const;

 private:
  class MetadataImpl;
  MetadataImpl const* impl_;
};

list* ReadDataCfg(char const* filename);
bool ReadOption(char* s, list* options);
void InsertOption(list* l, char* key, char* val);
char* FindOption(list* l, char* key);
char* FindOptionStr(list* l, char* key, char* def);
char* FindOptionStrQuiet(list* l, char* key, char* def);
int FindOptionInt(list* l, char* key, int def);
int FindOptionIntQuiet(list* l, char* key, int def);
float FindOptionFloat(list* l, char* key, float def);
float FindOptionFloatQuiet(list* l, char* key, float def);
void UnusedOption(list* l);
