#include "list.h"

#include <stdlib.h>
#include <string.h>

#include "utils.h"

list* MakeList()
{
  list* l = (list*)xmalloc(sizeof(list));
  l->size = 0;
  l->front = 0;
  l->back = 0;
  return l;
}

void InsertList(list* l, void* val)
{
  node* newnode = (node*)xmalloc(sizeof(node));
  newnode->val = val;
  newnode->next = 0;

  if (!l->back)
  {
    l->front = newnode;
    newnode->prev = 0;
  }
  else
  {
    l->back->next = newnode;
    newnode->prev = l->back;
  }
  l->back = newnode;
  ++l->size;
}

void FreeNode(node* n)
{
  node* next;
  while (n)
  {
    next = n->next;
    free(n);
    n = next;
  }
}

void FreeList(list* l)
{
  FreeNode(l->front);
  free(l);
}

void FreeListContents(list* l)
{
  node* n = l->front;
  while (n)
  {
    free(n->val);
    n = n->next;
  }
}

void FreeListContentsKvp(list* l)
{
  node* n = l->front;
  while (n)
  {
    kvp* p = (kvp*)n->val;
    free(p->key);
    free(n->val);
    n = n->next;
  }
}

void** ListToArray(list* l)
{
  void** a = (void**)xcalloc(l->size, sizeof(void*));
  int count = 0;
  node* n = l->front;
  while (n)
  {
    a[count++] = n->val;
    n = n->next;
  }
  return a;
}
