///////////////////////////////////////////////////////////////////////////////
// File name: Matrix.h
// This file defines the basic classes of vertex and edge, and the matrix data
// type is then defined based on them.
// In the matrix, the grids store the edges, which denote the utilities or
// costs in corresponding assignment matrix. There are two vectors of
// vertices, one for agents, and the other for tasks.
// Lantao Liu, Nov 1, 2009
// Last modified: 09/2011 -> MTL is removed and 2D vectors are used.
///////////////////////////////////////////////////////////////////////////////

#pragma once

#include <algorithm>
#include <string>
#include <vector>

#define TASKS_SIZE 3
#define AGENTS_SIZE 3
#define DOUBLE_EPSILON 1e-7

#define SEED 0
#define PERIOD 0
#define VERBOSE_LEVEL 1
#define PLOT 0

#define MAX_RANDOM 100

#define POS_INF 10e8
#define NEG_INF -10e8

typedef size_t VID;
typedef std::pair<size_t, size_t> EID;

class Edge;
class Vertex;

/////////////////////////////////////////////////////////////
//   Define the matrix type
//   Edge is the basi[c data type used in it
//   The weight of edge denotes the utility or cost
/////////////////////////////////////////////////////////////

typedef std::vector<std::vector<Edge> > Matrix;

////////////////////////////////////////////////////////////
//
//  Edge class: represent an element in matrix
//              or an edge in bipartite graph
//
////////////////////////////////////////////////////////////
class Edge
{
 public:
  Edge()
  {
    weight = 0;
    matched_flag = false;     // unmatched
    admissible_flag = false;  // inadmissible
  }

  Edge(EID _eid) : eid(_eid)
  {
    weight = 0;
    matched_flag = false;     // unmatched
    admissible_flag = false;  // inadmissible
  }
  ~Edge() {}

  EID GetEID(void) { return eid; }
  void SetEID(EID _eid) { eid = _eid; }

  double GetWeight(void) { return weight; }
  void SetWeight(double _weight) { weight = _weight; }

  bool GetMatchedFlag(void) { return matched_flag; }
  void SetMatchedFlag(bool _matched_flag) { matched_flag = _matched_flag; }

  bool GetAdmissibleFlag(void) { return admissible_flag; }
  void SetAdmissibleFlag(bool _admissible) { admissible_flag = _admissible; }

 private:
  // data members describing properties of an edge
  EID eid;
  double weight;
  bool matched_flag;
  bool admissible_flag;
};

///////////////////////////////////////////////////////////
//
//  Vertex class: represent an agent or a task
//
//////////////////////////////////////////////////////////

class Vertex
{
 public:
  Vertex() : label(0), matched(false), colored(false) {}
  Vertex(VID _vid) : vid(_vid), label(0), matched(false), colored(false) {}
  ~Vertex() {}

  VID GetVID(void) { return vid; }
  void SetVID(VID _vid) { vid = _vid; }

  std::string GetObj(void) { return obj; }
  void SetObj(std::string _obj) { obj = _obj; }

  double GetLabel(void) { return label; }
  void SetLabel(double _label) { label = _label; }

  bool GetMatched(void) { return matched; }
  void SetMatched(bool _matched) { matched = _matched; }

  bool GetColored(void) { return colored; }
  void SetColored(bool _colored) { colored = _colored; }

  bool GetVisited(void) { return visited; }
  void SetVisited(bool _visited) { visited = _visited; }

  std::vector<EID>* GetPath(void) { return &path; }

 private:
  // data members describing properties of a vertex
  VID vid;
  std::string obj;
  double label;
  bool matched;
  bool colored;  // colored if in the set of T or S
  bool visited;  // to flag if visited when go through alternating tree

 public:
  std::vector<EID> path;  // record previous path so far
};