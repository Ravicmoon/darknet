#pragma once
#include <string>
#include <vector>

#include "libapi.h"
#include "list.h"

class LIB_API Metadata
{
 public:
  Metadata();
  Metadata(std::string filename);
  ~Metadata();

  bool Get(std::string filename);
  int NumClasses() const;

  std::string TrainFile() const;
  std::string ValFile() const;
  std::string NameFile() const;
  std::string SaveDir() const;

  std::vector<std::string> TrainImgList() const;
  std::vector<std::string> ValImgList() const;
  std::vector<std::string> NameList() const;

 private:
  class MetadataImpl;
  MetadataImpl* impl_;
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
