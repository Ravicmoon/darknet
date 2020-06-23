#include "option_list.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <fstream>
#include <vector>

#include "data.h"
#include "utils.h"

class Metadata::MetadataImpl
{
 public:
  MetadataImpl();
  MetadataImpl(char const* filename);

  void Get(char const* filename);

 public:
  int classes_;

  std::string train_file_;
  std::string val_file_;
  std::string name_file_;
  std::string save_dir_;

  std::vector<std::string> train_img_list_;
  std::vector<std::string> val_img_list_;
  std::vector<std::string> name_list_;
};

Metadata::MetadataImpl::MetadataImpl() : classes_(0) {}

Metadata::MetadataImpl::MetadataImpl(char const* filename) { Get(filename); }

void Metadata::MetadataImpl::Get(char const* filename)
{
  list* options = ReadDataCfg(filename);

  classes_ = FindOptionInt(options, "classes", 2);

  train_file_ = FindOptionStr(options, "train", "train.txt");
  val_file_ = FindOptionStr(options, "valid", "valid.txt");
  name_file_ = FindOptionStr(options, "name", "name.txt");
  save_dir_ = FindOptionStr(options, "save", "save");

  if (Exists(train_file_.c_str()))
    train_img_list_ = GetList(train_file_);

  if (Exists(val_file_.c_str()))
    val_img_list_ = GetList(val_file_);

  if (Exists(name_file_.c_str()))
    name_list_ = GetList(name_file_);

  if ((int)name_list_.size() != classes_)
  {
    printf("Invalid metadata file: %d != %d", (int)name_list_.size(), classes_);
    exit(EXIT_FAILURE);
  }

  FreeList(options);
}

Metadata::Metadata() : impl_(new MetadataImpl()) {}

Metadata::Metadata(char const* filename) : impl_(new MetadataImpl(filename)) {}

Metadata::~Metadata() { delete impl_; }

void Metadata::Get(char const* filename) { impl_->Get(filename); }

int Metadata::NumClasses() const { return impl_->classes_; }

std::string Metadata::TrainFile() const { return impl_->train_file_; }
std::string Metadata::ValFile() const { return impl_->val_file_; }
std::string Metadata::NameFile() const { return impl_->name_file_; }
std::string Metadata::SaveDir() const { return impl_->save_dir_; }

std::vector<std::string> Metadata::TrainImgList() const
{
  return impl_->train_img_list_;
}
std::vector<std::string> Metadata::ValImgList() const
{
  return impl_->val_img_list_;
}
std::vector<std::string> Metadata::NameList() const
{
  return impl_->name_list_;
}

list* ReadDataCfg(char const* filename)
{
  FILE* file = fopen(filename, "r");
  if (file == nullptr)
    FileError(filename);

  char* line;
  int num_line = 0;
  list* options = MakeList();
  while ((line = fgetl(file)) != 0)
  {
    num_line++;
    strip(line);
    switch (line[0])
    {
      case '\0':
      case '#':
      case ';':
        free(line);
        break;
      default:
        if (!ReadOption(line, options))
        {
          printf(
              "Config file error line %d, could parse: %s\n", num_line, line);
          free(line);
        }
        break;
    }
  }
  fclose(file);

  return options;
}

bool ReadOption(char* s, list* options)
{
  size_t i;
  size_t len = strlen(s);
  char* val = 0;
  for (i = 0; i < len; ++i)
  {
    if (s[i] == '=')
    {
      s[i] = '\0';
      val = s + i + 1;
      break;
    }
  }
  if (i == len - 1)
    return false;

  InsertOption(options, s, val);
  return true;
}

void InsertOption(list* l, char* key, char* val)
{
  kvp* p = (kvp*)xmalloc(sizeof(kvp));
  p->key = key;
  p->val = val;
  p->used = 0;
  InsertList(l, p);
}

char* FindOption(list* l, char* key)
{
  node* n = l->front;
  while (n)
  {
    kvp* p = (kvp*)n->val;
    if (strcmp(p->key, key) == 0)
    {
      p->used = 1;
      return p->val;
    }
    n = n->next;
  }
  return 0;
}

char* FindOptionStr(list* l, char* key, char* def)
{
  char* v = FindOption(l, key);
  if (v)
    return v;
  if (def)
    fprintf(stderr, "%s: Using default '%s'\n", key, def);
  return def;
}

char* FindOptionStrQuiet(list* l, char* key, char* def)
{
  char* v = FindOption(l, key);
  if (v)
    return v;
  return def;
}

int FindOptionInt(list* l, char* key, int def)
{
  char* v = FindOption(l, key);
  if (v)
    return atoi(v);
  fprintf(stderr, "%s: Using default '%d'\n", key, def);
  return def;
}

int FindOptionIntQuiet(list* l, char* key, int def)
{
  char* v = FindOption(l, key);
  if (v)
    return atoi(v);
  return def;
}

float FindOptionFloatQuiet(list* l, char* key, float def)
{
  char* v = FindOption(l, key);
  if (v)
    return atof(v);
  return def;
}

float FindOptionFloat(list* l, char* key, float def)
{
  char* v = FindOption(l, key);
  if (v)
    return atof(v);
  fprintf(stderr, "%s: Using default '%lf'\n", key, def);
  return def;
}

void UnusedOption(list* l)
{
  node* n = l->front;
  while (n)
  {
    kvp* p = (kvp*)n->val;
    if (!p->used)
    {
      fprintf(stderr, "Unused field: '%s = %s'\n", p->key, p->val);
    }
    n = n->next;
  }
}