#pragma once

typedef struct
{
  char* key;
  char* val;
  int used;
} kvp;

typedef struct node
{
  void* val;
  struct node* next;
  struct node* prev;
} node;

typedef struct list
{
  int size;
  node* front;
  node* back;
} list;

list* MakeList();
void InsertList(list*, void*);
void FreeList(list* l);
void FreeListContents(list* l);
void FreeListContentsKvp(list* l);
void** ListToArray(list* l);
