#include "option_list.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "data.h"
#include "utils.h"

Metadata GetMetadata(char const* filename)
{
  Metadata m = {0};
  list* options = ReadDataCfg(filename);

  char* name_list = FindOptionStr(options, "names", 0);
  if (!name_list)
    name_list = FindOptionStr(options, "labels", 0);

  if (!name_list)
    fprintf(stderr, "No names or labels found\n");
  else
    m.names = GetLabels(name_list);

  m.classes = FindOptionInt(options, "classes", 2);

  free_list(options);

  if (name_list)
    printf("Loaded - names_list: %s, classes = %d \n", name_list, m.classes);

  return m;
}

list* ReadDataCfg(char const* filename)
{
  FILE* file = fopen(filename, "r");
  if (file == 0)
    FileError(filename);
  char* line;
  int nu = 0;
  list* options = MakeList();
  while ((line = fgetl(file)) != 0)
  {
    ++nu;
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
          fprintf(
              stderr, "Config file error line %d, could parse: %s\n", nu, line);
          free(line);
        }
        break;
    }
  }
  fclose(file);
  return options;
}

int ReadOption(char* s, list* options)
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
    return 0;
  char* key = s;
  InsertOption(options, key, val);
  return 1;
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