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
  MetadataImpl(char const* filename);

 public:
  int classes_;
  std::vector<std::string> names_;
};

Metadata::MetadataImpl::MetadataImpl(char const* filename)
{
  list* options = ReadDataCfg(filename);

  char* name_file = FindOptionStr(options, "names", 0);
  if (name_file == nullptr)
  {
    printf("Invalid metadata file: name file not found");
    exit(EXIT_FAILURE);
  }

  classes_ = FindOptionInt(options, "classes", 2);
  names_.clear();

  std::ifstream instream(name_file);
  while (instream.is_open() && !instream.eof())
  {
    char buffer[256];
    instream.getline(buffer, sizeof(buffer));

    if (std::string(buffer).empty())
      break;

    names_.push_back(buffer);
  }
  instream.close();

  if ((int)names_.size() != classes_)
  {
    printf("Invalid metadata file: %d != %d", (int)names_.size(), classes_);
    exit(EXIT_FAILURE);
  }

  FreeList(options);
}

Metadata::Metadata() : impl_(nullptr) {}

Metadata::Metadata(char const* filename) : impl_(new MetadataImpl(filename)) {}

Metadata::~Metadata()
{
  if (impl_ != nullptr)
    delete impl_;
}

void Metadata::Get(char const* filename)
{
  if (impl_ == nullptr)
    impl_ = new MetadataImpl(filename);
}

int Metadata::NumClasses() const
{
  if (impl_ != nullptr)
    return impl_->classes_;
  else
    return -1;
}

std::string Metadata::NameAt(int idx) const
{
  if (impl_ != nullptr)
    return impl_->names_[idx];
  else
    return std::string();
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