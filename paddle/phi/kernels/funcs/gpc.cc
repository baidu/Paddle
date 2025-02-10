// Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

/**
 * @file src/gpc.cpp
 * @author huhan02(com@baidu.com)
 * @date 2015/12/18 14:17:30
 * @brief
 *
 * @modified by sunyipeng
 * @email sunyipeng@baidu.com
 * @date 2018/6/12
 **/

#include "paddle/phi/kernels/funcs/gpc.h"
#include <array>

#include "paddle/phi/core/enforce.h"

namespace phi::funcs {

typedef struct lmt_shape { /* Local minima table                */
  double y;                /* Y coordinate at local minimum     */
  edge_node *first_bound;  /* Pointer to bound list             */
  struct lmt_shape *next;  /* Pointer to next local minimum     */
} lmt_node;

typedef struct sbt_t_shape { /* Scanbeam tree                     */
  double y;                  /* Scanbeam node y value             */
  struct sbt_t_shape *less;  /* Pointer to nodes with lower y     */
  struct sbt_t_shape *more;  /* Pointer to nodes with higher y    */
} sb_tree;

typedef struct it_shape {        /* Intersection table                */
  std::array<edge_node *, 2> ie; /* Intersecting edge (bundle) pair   */
  gpc_vertex point;              /* Point of intersection             */
  struct it_shape *next;         /* The next intersection table node  */
} it_node;

typedef struct st_shape { /* Sorted edge table                 */
  edge_node *edge;        /* Pointer to AET edge               */
  double xb;              /* Scanbeam bottom x coordinate      */
  double xt;              /* Scanbeam top x coordinate         */
  double dx;              /* Change in x for a unit y increase */
  struct st_shape *prev;  /* Previous edge in sorted list      */
} st_node;

typedef struct bbox_shape { /* Contour axis-aligned bounding box */
  double xmin;              /* Minimum x coordinate              */
  double ymin;              /* Minimum y coordinate              */
  double xmax;              /* Maximum x coordinate              */
  double ymax;              /* Maximum y coordinate              */
} bbox;

/*
===========================================================================
                               Global Data
===========================================================================
*/

/* Horizontal edge state transitions within scanbeam boundary */
const std::array<std::array<h_state, 6>, 3> next_h_state = {
    {/*        ABOVE     BELOW     CROSS */
     /*        L   R     L   R     L   R */
     /* NH */
     {{BH, TH, TH, BH, NH, NH}},
     /* BH */
     {{NH, NH, NH, NH, TH, TH}},
     /* TH */
     {NH, NH, NH, NH, BH, BH}}};
/*
===========================================================================
                             Private Functions
===========================================================================
*/

static void reset_it(it_node **it) {
  it_node *itn = nullptr;

  while (*it) {
    itn = (*it)->next;
    gpc_free<it_node>(*it);
    *it = itn;
  }
}

static void reset_lmt(lmt_node **lmt) {
  lmt_node *lmtn = nullptr;

  while (*lmt) {
    lmtn = (*lmt)->next;
    gpc_free<lmt_node>(*lmt);
    *lmt = lmtn;
  }
}

static void insert_bound(edge_node **b, edge_node *e) {
  edge_node *existing_bound = nullptr;

  if (!*b) {
    /* Link node e to the tail of the list */
    *b = e;
  } else {
    /* Do primary sort on the x field */
    if (e[0].bot.x < (*b)[0].bot.x) {
      /* Insert a new node mid-list */
      existing_bound = *b;
      *b = e;
      (*b)->next_bound = existing_bound;
    } else {
      if (e[0].bot.x == (*b)[0].bot.x) {
        /* Do secondary sort on the dx field */
        if (e[0].dx < (*b)[0].dx) {
          /* Insert a new node mid-list */
          existing_bound = *b;
          *b = e;
          (*b)->next_bound = existing_bound;
        } else {
          /* Head further down the list */
          insert_bound(&((*b)->next_bound), e);
        }
      } else {
        /* Head further down the list */
        insert_bound(&((*b)->next_bound), e);
      }
    }
  }
}

static edge_node **bound_list(lmt_node **lmt, double y) {
  lmt_node *existing_node = nullptr;

  if (!*lmt) {
    /* Add node onto the tail end of the LMT */
    gpc_malloc<lmt_node>(
        *lmt, sizeof(lmt_node), const_cast<char *>("LMT insertion"));  // NOLINT
    (*lmt)->y = y;
    (*lmt)->first_bound = nullptr;
    (*lmt)->next = nullptr;
    return &((*lmt)->first_bound);
  } else if (y < (*lmt)->y) {
    /* Insert a new LMT node before the current node */
    existing_node = *lmt;
    gpc_malloc<lmt_node>(
        *lmt, sizeof(lmt_node), const_cast<char *>("LMT insertion"));  // NOLINT
    (*lmt)->y = y;
    (*lmt)->first_bound = nullptr;
    (*lmt)->next = existing_node;
    return &((*lmt)->first_bound);
  } else {
    if (y > (*lmt)->y) {
      /* Head further up the LMT */
      return bound_list(&((*lmt)->next), y);
    } else {
      /* Use this existing LMT node */
      return &((*lmt)->first_bound);
    }
  }
}

static void add_to_sbtree(int *entries, sb_tree **sbtree, double y) {
  if (!*sbtree) {
    /* Add a new tree node here */
    gpc_malloc<sb_tree>(
        *sbtree,
        sizeof(sb_tree),
        const_cast<char *>("scanbeam tree insertion"));  // NOLINT
    (*sbtree)->y = y;
    (*sbtree)->less = nullptr;
    (*sbtree)->more = nullptr;
    (*entries)++;
  } else {
    if ((*sbtree)->y > y) {
      /* Head into the 'less' sub-tree */
      add_to_sbtree(entries, &((*sbtree)->less), y);
    } else {
      if ((*sbtree)->y < y) {
        /* Head into the 'more' sub-tree */
        add_to_sbtree(entries, &((*sbtree)->more), y);
      }
    }
  }
}

static void build_sbt(int *entries, double *sbt, sb_tree *sbtree) {
  if (sbtree->less) {
    build_sbt(entries, sbt, sbtree->less);
  }
  sbt[*entries] = sbtree->y;
  (*entries)++;
  if (sbtree->more) {
    build_sbt(entries, sbt, sbtree->more);
  }
}

static void free_sbtree(sb_tree **sbtree) {
  if (*sbtree) {
    free_sbtree(&((*sbtree)->less));
    free_sbtree(&((*sbtree)->more));
    gpc_free<sb_tree>(*sbtree);
  }
}

static int count_optimal_vertices(gpc_vertex_list c) {
  int result = 0;
  int i = 0;

  /* Ignore non-contributing contours */
  if (c.num_vertices > 0) {
    for (i = 0; i < c.num_vertices; i++) {
      /* Ignore superfluous vertices embedded in horizontal edges */
      if (gpc_optimal(c.vertex, i, c.num_vertices)) {
        result++;
      }
    }
  }
  return result;
}

static edge_node *build_lmt(lmt_node **lmt,
                            sb_tree **sbtree,
                            int *sbt_entries,
                            gpc_polygon *p,
                            int type,
                            gpc_op op) {
  int c = 0;
  int i = 0;
  int min = 0;
  int max = 0;
  int num_edges = 0;
  int v = 0;
  int num_vertices = 0;
  int total_vertices = 0;
  int e_index = 0;
  edge_node *e = nullptr;
  edge_node *edge_table = nullptr;

  for (c = 0; c < p->num_contours; c++) {
    total_vertices += count_optimal_vertices(p->contour[c]);
  }

  /* Create the entire input polygon edge table in one go */
  gpc_malloc<edge_node>(edge_table,
                        total_vertices * static_cast<int>(sizeof(edge_node)),
                        const_cast<char *>("edge table creation"));  // NOLINT

  for (c = 0; c < p->num_contours; c++) {
    if (p->contour[c].num_vertices < 0) {
      /* Ignore the non-contributing contour and repair the vertex count */
      p->contour[c].num_vertices = -p->contour[c].num_vertices;
    } else {
      /* Perform contour optimisation */
      num_vertices = 0;
      for (i = 0; i < p->contour[c].num_vertices; i++) {
        if (gpc_optimal(p->contour[c].vertex, i, p->contour[c].num_vertices)) {
          edge_table[num_vertices].vertex.x = p->contour[c].vertex[i].x;
          edge_table[num_vertices].vertex.y = p->contour[c].vertex[i].y;

          /* Record vertex in the scanbeam table */
          add_to_sbtree(sbt_entries, sbtree, edge_table[num_vertices].vertex.y);

          num_vertices++;
        }
      }

      /* Do the contour forward pass */
      for (min = 0; min < num_vertices; min++) {
        /* If a forward local minimum... */
        if (gpc_fwd_min(edge_table, min, num_vertices)) {
          /* Search for the next local maximum... */
          num_edges = 1;
          max = gpc_next_index(min, num_vertices);
          while (gpc_not_fmax(edge_table, max, num_vertices)) {
            num_edges++;
            max = gpc_next_index(max, num_vertices);
          }

          /* Build the next edge list */
          e = &edge_table[e_index];
          e_index += num_edges;
          v = min;
          e[0].bstate[BELOW] = UNBUNDLED;
          e[0].bundle[BELOW][CLIP] = 0;
          e[0].bundle[BELOW][SUBJ] = 0;
          for (i = 0; i < num_edges; i++) {
            e[i].xb = edge_table[v].vertex.x;
            e[i].bot.x = edge_table[v].vertex.x;
            e[i].bot.y = edge_table[v].vertex.y;

            v = gpc_next_index(v, num_vertices);

            e[i].top.x = edge_table[v].vertex.x;
            e[i].top.y = edge_table[v].vertex.y;
            e[i].dx = (edge_table[v].vertex.x - e[i].bot.x) /
                      (e[i].top.y - e[i].bot.y);
            e[i].type = type;
            e[i].outp[ABOVE] = nullptr;
            e[i].outp[BELOW] = nullptr;
            e[i].next = nullptr;
            e[i].prev = nullptr;
            e[i].succ = ((num_edges > 1) && (i < (num_edges - 1))) ? &(e[i + 1])
                                                                   : nullptr;
            e[i].pred = ((num_edges > 1) && (i > 0)) ? &(e[i - 1]) : nullptr;
            e[i].next_bound = nullptr;
            e[i].bside[CLIP] = (op == GPC_DIFF) ? RIGHT : LEFT;
            e[i].bside[SUBJ] = LEFT;
          }
          insert_bound(bound_list(lmt, edge_table[min].vertex.y), e);
        }
      }

      /* Do the contour reverse pass */
      for (min = 0; min < num_vertices; min++) {
        /* If a reverse local minimum... */
        if (gpc_rev_min(edge_table, min, num_vertices)) {
          /* Search for the previous local maximum... */
          num_edges = 1;
          max = gpc_prev_index(min, num_vertices);
          while (gpc_not_rmax(edge_table, max, num_vertices)) {
            num_edges++;
            max = gpc_prev_index(max, num_vertices);
          }

          /* Build the previous edge list */
          e = &edge_table[e_index];
          e_index += num_edges;
          v = min;
          e[0].bstate[BELOW] = UNBUNDLED;
          e[0].bundle[BELOW][CLIP] = 0;
          e[0].bundle[BELOW][SUBJ] = 0;
          for (i = 0; i < num_edges; i++) {
            e[i].xb = edge_table[v].vertex.x;
            e[i].bot.x = edge_table[v].vertex.x;
            e[i].bot.y = edge_table[v].vertex.y;

            v = gpc_prev_index(v, num_vertices);

            e[i].top.x = edge_table[v].vertex.x;
            e[i].top.y = edge_table[v].vertex.y;
            e[i].dx = (edge_table[v].vertex.x - e[i].bot.x) /
                      (e[i].top.y - e[i].bot.y);
            e[i].type = type;
            e[i].outp[ABOVE] = nullptr;
            e[i].outp[BELOW] = nullptr;
            e[i].next = nullptr;
            e[i].prev = nullptr;
            e[i].succ = ((num_edges > 1) && (i < (num_edges - 1))) ? &(e[i + 1])
                                                                   : nullptr;
            e[i].pred = ((num_edges > 1) && (i > 0)) ? &(e[i - 1]) : nullptr;
            e[i].next_bound = nullptr;
            e[i].bside[CLIP] = (op == GPC_DIFF) ? RIGHT : LEFT;
            e[i].bside[SUBJ] = LEFT;
          }
          insert_bound(bound_list(lmt, edge_table[min].vertex.y), e);
        }
      }
    }
  }
  return edge_table;
}  // NOLINT

static void add_edge_to_aet(edge_node **aet, edge_node *edge, edge_node *prev) {
  if (!*aet) {
    /* Append edge onto the tail end of the AET */
    *aet = edge;
    edge->prev = prev;
    edge->next = nullptr;
  } else {
    /* Do primary sort on the xb field */
    if (edge->xb < (*aet)->xb) {
      /* Insert edge here (before the AET edge) */
      edge->prev = prev;
      edge->next = *aet;
      (*aet)->prev = edge;
      *aet = edge;
    } else {
      if (edge->xb == (*aet)->xb) {
        /* Do secondary sort on the dx field */
        if (edge->dx < (*aet)->dx) {
          /* Insert edge here (before the AET edge) */
          edge->prev = prev;
          edge->next = *aet;
          (*aet)->prev = edge;
          *aet = edge;
        } else {
          /* Head further into the AET */
          add_edge_to_aet(&((*aet)->next), edge, *aet);
        }
      } else {
        /* Head further into the AET */
        add_edge_to_aet(&((*aet)->next), edge, *aet);
      }
    }
  }
}

static void add_intersection(
    it_node **it, edge_node *edge0, edge_node *edge1, double x, double y) {
  it_node *existing_node = nullptr;

  if (!*it) {
    /* Append a new node to the tail of the list */
    gpc_malloc<it_node>(
        *it, sizeof(it_node), const_cast<char *>("IT insertion"));  // NOLINT
    (*it)->ie[0] = edge0;
    (*it)->ie[1] = edge1;
    (*it)->point.x = x;
    (*it)->point.y = y;
    (*it)->next = nullptr;
  } else {
    if ((*it)->point.y > y) {
      /* Insert a new node mid-list */
      existing_node = *it;
      gpc_malloc<it_node>(
          *it, sizeof(it_node), const_cast<char *>("IT insertion"));  // NOLINT
      (*it)->ie[0] = edge0;
      (*it)->ie[1] = edge1;
      (*it)->point.x = x;
      (*it)->point.y = y;
      (*it)->next = existing_node;
    } else {
      /* Head further down the list */
      add_intersection(&((*it)->next), edge0, edge1, x, y);
    }
  }
}

static void add_st_edge(st_node **st,
                        it_node **it,
                        edge_node *edge,
                        double dy) {
  st_node *existing_node = nullptr;
  double den = 0.0;
  double r = 0.0;
  double x = 0.0;
  double y = 0.0;

  if (!*st) {
    /* Append edge onto the tail end of the ST */
    gpc_malloc<st_node>(
        *st, sizeof(st_node), const_cast<char *>("ST insertion"));  // NOLINT
    (*st)->edge = edge;
    (*st)->xb = edge->xb;
    (*st)->xt = edge->xt;
    (*st)->dx = edge->dx;
    (*st)->prev = nullptr;
  } else {
    den = ((*st)->xt - (*st)->xb) - (edge->xt - edge->xb);

    /* If new edge and ST edge don't cross */
    if ((edge->xt >= (*st)->xt) || (edge->dx == (*st)->dx) ||
        (fabs(den) <= DBL_EPSILON)) {
      /* No intersection - insert edge here (before the ST edge) */
      existing_node = *st;
      gpc_malloc<st_node>(
          *st, sizeof(st_node), const_cast<char *>("ST insertion"));  // NOLINT
      (*st)->edge = edge;
      (*st)->xb = edge->xb;
      (*st)->xt = edge->xt;
      (*st)->dx = edge->dx;
      (*st)->prev = existing_node;
    } else {
      /* Compute intersection between new edge and ST edge */
      r = (edge->xb - (*st)->xb) / den;
      x = (*st)->xb + r * ((*st)->xt - (*st)->xb);
      y = r * dy;

      /* Insert the edge pointers and the intersection point in the IT */
      add_intersection(it, (*st)->edge, edge, x, y);

      /* Head further into the ST */
      add_st_edge(&((*st)->prev), it, edge, dy);
    }
  }
}

static void build_intersection_table(it_node **it, edge_node *aet, double dy) {
  st_node *st = nullptr;
  st_node *stp = nullptr;
  edge_node *edge = nullptr;

  /* Build intersection table for the current scanbeam */
  reset_it(it);
  st = nullptr;

  /* Process each AET edge */
  for (edge = aet; edge; edge = edge->next) {
    if ((edge->bstate[ABOVE] == BUNDLE_HEAD) || edge->bundle[ABOVE][CLIP] ||
        edge->bundle[ABOVE][SUBJ]) {
      add_st_edge(&st, it, edge, dy);
    }
  }

  /* Free the sorted edge table */
  while (st) {
    stp = st->prev;
    gpc_free<st_node>(st);
    st = stp;
  }
}

static int count_contours(polygon_node *polygon) {
  int nc = 0;
  int nv = 0;
  vertex_node *v = nullptr;
  vertex_node *nextv = nullptr;

  for (nc = 0; polygon; polygon = polygon->next) {
    if (polygon->active) {
      /* Count the vertices in the current contour */
      nv = 0;
      for (v = polygon->proxy->v[LEFT]; v; v = v->next) {
        nv++;
      }

      /* Record valid vertex counts in the active field */
      if (nv > 2) {
        polygon->active = nv;
        nc++;
      } else {
        /* Invalid contour: just free the heap */
        for (v = polygon->proxy->v[LEFT]; v; v = nextv) {
          nextv = v->next;
          gpc_free<vertex_node>(v);
        }
        polygon->active = 0;
      }
    }
  }
  return nc;
}

static void add_left(polygon_node *p, double x, double y) {
  PADDLE_ENFORCE_NOT_NULL(
      p, common::errors::InvalidArgument("Input polygon node is nullptr."));
  vertex_node *nv = nullptr;

  /* Create a new vertex node and set its fields */
  gpc_malloc<vertex_node>(
      nv,
      sizeof(vertex_node),
      const_cast<char *>("vertex node creation"));  // NOLINT
  nv->x = x;
  nv->y = y;

  /* Add vertex nv to the left end of the polygon's vertex list */
  nv->next = p->proxy->v[LEFT];

  /* Update proxy->[LEFT] to point to nv */
  p->proxy->v[LEFT] = nv;
}

static void merge_left(polygon_node *p, polygon_node *q, polygon_node *list) {
  polygon_node *target = nullptr;

  /* Label contour as a hole */
  q->proxy->hole = 1;

  if (p->proxy != q->proxy) {
    /* Assign p's vertex list to the left end of q's list */
    p->proxy->v[RIGHT]->next = q->proxy->v[LEFT];
    q->proxy->v[LEFT] = p->proxy->v[LEFT];

    /* Redirect any p->proxy references to q->proxy */

    for (target = p->proxy; list; list = list->next) {
      if (list->proxy == target) {
        list->active = 0;
        list->proxy = q->proxy;
      }
    }
  }
}

static void add_right(polygon_node *p, double x, double y) {
  vertex_node *nv = nullptr;

  /* Create a new vertex node and set its fields */
  gpc_malloc<vertex_node>(
      nv,
      sizeof(vertex_node),
      const_cast<char *>("vertex node creation"));  // NOLINT
  nv->x = x;
  nv->y = y;
  nv->next = nullptr;

  /* Add vertex nv to the right end of the polygon's vertex list */
  p->proxy->v[RIGHT]->next = nv;

  /* Update proxy->v[RIGHT] to point to nv */
  p->proxy->v[RIGHT] = nv;
}

static void merge_right(polygon_node *p, polygon_node *q, polygon_node *list) {
  PADDLE_ENFORCE_NOT_NULL(
      p, common::errors::InvalidArgument("Input polygon node is nullptr."));
  polygon_node *target = nullptr;

  /* Label contour as external */
  q->proxy->hole = 0;

  if (p->proxy != q->proxy) {
    /* Assign p's vertex list to the right end of q's list */
    q->proxy->v[RIGHT]->next = p->proxy->v[LEFT];
    q->proxy->v[RIGHT] = p->proxy->v[RIGHT];

    /* Redirect any p->proxy references to q->proxy */
    for (target = p->proxy; list; list = list->next) {
      if (list->proxy == target) {
        list->active = 0;
        list->proxy = q->proxy;
      }
    }
  }
}

static void add_local_min(polygon_node **p,
                          edge_node *edge,
                          double x,
                          double y) {
  polygon_node *existing_min = nullptr;
  vertex_node *nv = nullptr;

  existing_min = *p;

  gpc_malloc<polygon_node>(
      *p,
      sizeof(polygon_node),
      const_cast<char *>("polygon node creation"));  // NOLINT

  /* Create a new vertex node and set its fields */
  gpc_malloc<vertex_node>(
      nv,
      sizeof(vertex_node),
      const_cast<char *>("vertex node creation"));  // NOLINT
  nv->x = x;
  nv->y = y;
  nv->next = nullptr;

  /* Initialise proxy to point to p itself */
  (*p)->proxy = (*p);
  (*p)->active = 1;
  (*p)->next = existing_min;

  /* Make v[LEFT] and v[RIGHT] point to new vertex nv */
  (*p)->v[LEFT] = nv;
  (*p)->v[RIGHT] = nv;

  /* Assign polygon p to the edge */
  edge->outp[ABOVE] = *p;
}

static int count_tristrips(polygon_node *tn) {
  int total = 0;

  for (total = 0; tn; tn = tn->next) {
    if (tn->active > 2) {
      total++;
    }
  }
  return total;
}

void add_vertex(vertex_node **t, double x, double y) {
  if (!(*t)) {
    gpc_malloc<vertex_node>(
        *t,
        sizeof(vertex_node),
        const_cast<char *>("tristrip vertex creation"));  // NOLINT
    (*t)->x = x;
    (*t)->y = y;
    (*t)->next = nullptr;
  } else {
    /* Head further down the list */
    add_vertex(&((*t)->next), x, y);
  }
}

void gpc_vertex_create(edge_node *e, int p, int s, double x, double y) {
  PADDLE_ENFORCE_NOT_NULL(
      e, common::errors::InvalidArgument("Input edge node is nullptr."));
  add_vertex(&(e->outp[p]->v[s]), x, y);
  e->outp[p]->active++;
}

static void new_tristrip(polygon_node **tn,
                         edge_node *edge,
                         double x,
                         double y) {
  if (!(*tn)) {
    gpc_malloc<polygon_node>(
        *tn,
        sizeof(polygon_node),
        const_cast<char *>("tristrip node creation"));  // NOLINT
    (*tn)->next = nullptr;
    (*tn)->v[LEFT] = nullptr;
    (*tn)->v[RIGHT] = nullptr;
    (*tn)->active = 1;
    add_vertex(&((*tn)->v[LEFT]), x, y);
    edge->outp[ABOVE] = *tn;
  } else {
    /* Head further down the list */
    new_tristrip(&((*tn)->next), edge, x, y);
  }
}

static bbox *create_contour_bboxes(gpc_polygon *p) {
  bbox *box = nullptr;
  int c = 0;
  int v = 0;

  gpc_malloc<bbox>(box,
                   p->num_contours * static_cast<int>(sizeof(bbox)),
                   const_cast<char *>("Bounding box creation"));  // NOLINT
  PADDLE_ENFORCE_NOT_NULL(
      box, common::errors::ResourceExhausted("Failed to malloc box memory."));

  /* Construct contour bounding boxes */
  for (c = 0; c < p->num_contours; c++) {
    /* Initialise bounding box extent */
    box[c].xmin = DBL_MAX;
    box[c].ymin = DBL_MAX;
    box[c].xmax = -DBL_MAX;
    box[c].ymax = -DBL_MAX;

    for (v = 0; v < p->contour[c].num_vertices; v++) {
      /* Adjust bounding box */
      if (p->contour[c].vertex[v].x < box[c].xmin) {
        box[c].xmin = p->contour[c].vertex[v].x;
      }
      if (p->contour[c].vertex[v].y < box[c].ymin) {
        box[c].ymin = p->contour[c].vertex[v].y;
      }
      if (p->contour[c].vertex[v].x > box[c].xmax) {
        box[c].xmax = p->contour[c].vertex[v].x;
      }
      if (p->contour[c].vertex[v].y > box[c].ymax) {
        box[c].ymax = p->contour[c].vertex[v].y;
      }
    }
  }
  return box;
}

static void minimax_test(gpc_polygon *subj, gpc_polygon *clip, gpc_op op) {
  bbox *s_bbox = nullptr;
  bbox *c_bbox = nullptr;
  int s = 0;
  int c = 0;
  int *o_table = nullptr;
  int overlap = 0;

  s_bbox = create_contour_bboxes(subj);
  c_bbox = create_contour_bboxes(clip);

  gpc_malloc<int>(
      o_table,
      subj->num_contours * clip->num_contours * static_cast<int>(sizeof(int)),
      const_cast<char *>("overlap table creation"));  // NOLINT

  /* Check all subject contour bounding boxes against clip boxes */
  for (s = 0; s < subj->num_contours; s++) {
    for (c = 0; c < clip->num_contours; c++) {
      o_table[c * subj->num_contours + s] =
          (!((s_bbox[s].xmax < c_bbox[c].xmin) ||
             (s_bbox[s].xmin > c_bbox[c].xmax))) &&
          (!((s_bbox[s].ymax < c_bbox[c].ymin) ||
             (s_bbox[s].ymin > c_bbox[c].ymax)));
    }
  }

  /* For each clip contour, search for any subject contour overlaps */
  for (c = 0; c < clip->num_contours; c++) {
    overlap = 0;
    for (s = 0; (!overlap) && (s < subj->num_contours); s++) {
      overlap = o_table[c * subj->num_contours + s];
    }

    if (!overlap) {
      /* Flag non contributing status by negating vertex count */
      clip->contour[c].num_vertices = -clip->contour[c].num_vertices;
    }
  }

  if (op == GPC_INT) {
    /* For each subject contour, search for any clip contour overlaps */
    for (s = 0; s < subj->num_contours; s++) {
      overlap = 0;
      for (c = 0; (!overlap) && (c < clip->num_contours); c++) {
        overlap = o_table[c * subj->num_contours + s];
      }

      if (!overlap) {
        /* Flag non contributing status by negating vertex count */
        subj->contour[s].num_vertices = -subj->contour[s].num_vertices;
      }
    }
  }

  gpc_free<bbox>(s_bbox);
  gpc_free<bbox>(c_bbox);
  gpc_free<int>(o_table);
}

/*
===========================================================================
                             Public Functions
===========================================================================
*/

void gpc_free_polygon(gpc_polygon *p) {
  int c = 0;

  for (c = 0; c < p->num_contours; c++) {
    gpc_free<gpc_vertex>(p->contour[c].vertex);
  }
  gpc_free<int>(p->hole);
  gpc_free<gpc_vertex_list>(p->contour);
  p->num_contours = 0;
}

/*
void gpc_read_polygon(FILE *fp, int read_hole_flags, gpc_polygon *p) {
  int c = 0;
  int v = 0;

  fscanf(fp, "%d", &(p->num_contours));
  gpc_malloc<int>(p->hole, p->num_contours * sizeof(int),
                  (char *)"hole flag array creation");
  gpc_malloc<gpc_vertex_list>(p->contour,
                              p->num_contours * sizeof(gpc_vertex_list),
                              (char *)"contour creation");
  for (c = 0; c < p->num_contours; c++) {
    fscanf(fp, "%d", &(p->contour[c].num_vertices));

    if (read_hole_flags) {
      fscanf(fp, "%d", &(p->hole[c]));
    } else {
      p->hole[c] = 0; // Assume all contours to be external
    }

    gpc_malloc<gpc_vertex>(p->contour[c].vertex,
                           p->contour[c].num_vertices * sizeof(gpc_vertex),
                           (char *)"vertex creation");
    for (v = 0; v < p->contour[c].num_vertices; v++) {
      fscanf(fp, "%lf %lf", &(p->contour[c].vertex[v].x),
             &(p->contour[c].vertex[v].y));
    }
  }
}

void gpc_write_polygon(FILE *fp, int write_hole_flags, gpc_polygon *p) {
  int c = 0;
  int v = 0;

  fprintf(fp, "%d\n", p->num_contours);
  for (c = 0; c < p->num_contours; c++) {
    fprintf(fp, "%d\n", p->contour[c].num_vertices);

    if (write_hole_flags) {
      fprintf(fp, "%d\n", p->hole[c]);
    }

    for (v = 0; v < p->contour[c].num_vertices; v++) {
      fprintf(fp, "% .*lf % .*lf\n", DBL_DIG, p->contour[c].vertex[v].x,
              DBL_DIG, p->contour[c].vertex[v].y);
    }
  }
}
*/

void gpc_add_contour(gpc_polygon *p, gpc_vertex_list *new_contour, int hole) {
  int *extended_hole = nullptr;
  int c = 0;
  int v = 0;
  gpc_vertex_list *extended_contour = nullptr;

  /* Create an extended hole array */
  gpc_malloc<int>(extended_hole,
                  (p->num_contours + 1) * static_cast<int>(sizeof(int)),
                  const_cast<char *>("contour hole addition"));  // NOLINT
  PADDLE_ENFORCE_NOT_NULL(extended_hole,
                          common::errors::ResourceExhausted(
                              "Failed to malloc extended hole memory."));

  /* Create an extended contour array */
  gpc_malloc<gpc_vertex_list>(
      extended_contour,
      (p->num_contours + 1) * static_cast<int>(sizeof(gpc_vertex_list)),
      const_cast<char *>("contour addition"));  // NOLINT

  /* Copy the old contour and hole data into the extended arrays */
  for (c = 0; c < p->num_contours; c++) {
    extended_hole[c] = p->hole[c];
    extended_contour[c] = p->contour[c];  // NOLINT
  }

  /* Copy the new contour and hole onto the end of the extended arrays */
  c = p->num_contours;
  extended_hole[c] = hole;
  extended_contour[c].num_vertices = new_contour->num_vertices;
  gpc_malloc<gpc_vertex>(
      extended_contour[c].vertex,
      new_contour->num_vertices * static_cast<int>(sizeof(gpc_vertex)),
      const_cast<char *>("contour addition"));  // NOLINT
  for (v = 0; v < new_contour->num_vertices; v++) {
    extended_contour[c].vertex[v] = new_contour->vertex[v];  // NOLINT
  }

  /* Dispose of the old contour */
  gpc_free<gpc_vertex_list>(p->contour);
  gpc_free<int>(p->hole);

  /* Update the polygon information */
  p->num_contours++;
  p->hole = extended_hole;
  p->contour = extended_contour;
}

// gpc_polygon_clip
void gpc_polygon_clip(gpc_op op,
                      gpc_polygon *subj,
                      gpc_polygon *clip,
                      gpc_polygon *result) {
  sb_tree *sbtree = nullptr;
  it_node *it = nullptr;
  it_node *intersect = nullptr;
  edge_node *edge = nullptr;
  edge_node *prev_edge = nullptr;
  edge_node *next_edge = nullptr;
  edge_node *succ_edge = nullptr;
  edge_node *e0 = nullptr;
  edge_node *e1 = nullptr;
  edge_node *aet = nullptr;
  edge_node *c_heap = nullptr;
  edge_node *s_heap = nullptr;
  lmt_node *lmt = nullptr;
  lmt_node *local_min = nullptr;
  polygon_node *out_poly = nullptr;
  polygon_node *p = nullptr;
  polygon_node *q = nullptr;
  polygon_node *poly = nullptr;
  polygon_node *npoly = nullptr;
  polygon_node *cf = nullptr;
  vertex_node *vtx = nullptr;
  vertex_node *nv = nullptr;
  std::array<h_state, 2> horiz = {};
  std::array<int, 2> in = {};
  std::array<int, 2> exists = {};
  std::array<int, 2> parity = {LEFT, LEFT};
  int c = 0;
  int v = 0;
  int contributing = 0;
  int search = 0;
  int scanbeam = 0;
  int sbt_entries = 0;
  int vclass = 0;
  int bl = 0;
  int br = 0;
  int tl = 0;
  int tr = 0;
  double *sbt = nullptr;
  double xb = 0.0;
  double px = 0.0;
  double yb = 0.0;
  double yt = 0.0;
  double dy = 0.0;
  double ix = 0.0;
  double iy = 0.0;

  /* Test for trivial NULL result cases */
  if (((subj->num_contours == 0) && (clip->num_contours == 0)) ||
      ((subj->num_contours == 0) && ((op == GPC_INT) || (op == GPC_DIFF))) ||
      ((clip->num_contours == 0) && (op == GPC_INT))) {
    result->num_contours = 0;
    result->hole = nullptr;
    result->contour = nullptr;
    return;
  }
  /* Identify potentially contributing contours */
  if (((op == GPC_INT) || (op == GPC_DIFF)) && (subj->num_contours > 0) &&
      (clip->num_contours > 0)) {
    minimax_test(subj, clip, op);
  }
  /* Build LMT */
  if (subj->num_contours > 0) {
    s_heap = build_lmt(&lmt, &sbtree, &sbt_entries, subj, SUBJ, op);
  }
  if (clip->num_contours > 0) {
    c_heap = build_lmt(&lmt, &sbtree, &sbt_entries, clip, CLIP, op);
  }
  /* Return a NULL result if no contours contribute */
  if (lmt == nullptr) {
    result->num_contours = 0;
    result->hole = nullptr;
    result->contour = nullptr;
    reset_lmt(&lmt);
    gpc_free<edge_node>(s_heap);
    gpc_free<edge_node>(c_heap);
    return;
  }

  /* Build scanbeam table from scanbeam tree */
  gpc_malloc<double>(sbt,
                     sbt_entries * static_cast<int>(sizeof(double)),
                     const_cast<char *>("sbt creation"));  // NOLINT
  PADDLE_ENFORCE_NOT_NULL(sbt,
                          common::errors::ResourceExhausted(
                              "Failed to malloc scanbeam table memory."));

  build_sbt(&scanbeam, sbt, sbtree);
  scanbeam = 0;
  free_sbtree(&sbtree);
  /* Allow pointer re-use without causing memory leak */
  if (subj == result) {
    gpc_free_polygon(subj);
  }
  if (clip == result) {
    gpc_free_polygon(clip);
  }
  /* Invert clip polygon for difference operation */
  if (op == GPC_DIFF) {
    parity[CLIP] = RIGHT;
  }
  local_min = lmt;

  // Process each scanbeam
  while (scanbeam < sbt_entries) {
    /* Set yb and yt to the bottom and top of the scanbeam */
    yb = sbt[scanbeam++];
    if (scanbeam < sbt_entries) {
      yt = sbt[scanbeam];  // NOLINT
      dy = yt - yb;
    }
    /* === SCANBEAM BOUNDARY PROCESSING ================================ */
    /* If LMT node corresponding to yb exists */
    if (local_min) {
      if (local_min->y == yb) {
        /* Add edges starting at this local minimum to the AET */
        for (edge = local_min->first_bound; edge; edge = edge->next_bound) {
          add_edge_to_aet(&aet, edge, nullptr);
        }
        local_min = local_min->next;
      }
    }
    /* Set dummy previous x value */
    px = -DBL_MAX;
    /* Create bundles within AET */
    e0 = aet;
    e1 = aet;  // NOLINT
    /* Set up bundle fields of first edge */
    PADDLE_ENFORCE_NOT_NULL(
        aet, common::errors::InvalidArgument("Edge node AET is nullptr."));

    aet->bundle[ABOVE][aet->type] = (aet->top.y != yb);
    aet->bundle[ABOVE][!aet->type] = 0;
    aet->bstate[ABOVE] = UNBUNDLED;

    for (next_edge = aet->next; next_edge; next_edge = next_edge->next) {
      /* Set up bundle fields of next edge */
      next_edge->bundle[ABOVE][next_edge->type] = (next_edge->top.y != yb);
      next_edge->bundle[ABOVE][!next_edge->type] = 0;
      next_edge->bstate[ABOVE] = UNBUNDLED;
      /* Bundle edges above the scanbeam boundary if they coincide */
      if (next_edge->bundle[ABOVE][next_edge->type]) {
        if (gpc_eq(e0->xb, next_edge->xb) && gpc_eq(e0->dx, next_edge->dx) &&
            (e0->top.y != yb)) {
          next_edge->bundle[ABOVE][next_edge->type] ^=
              e0->bundle[ABOVE][next_edge->type];
          next_edge->bundle[ABOVE][!next_edge->type] =
              e0->bundle[ABOVE][!next_edge->type];
          next_edge->bstate[ABOVE] = BUNDLE_HEAD;
          e0->bundle[ABOVE][CLIP] = 0;
          e0->bundle[ABOVE][SUBJ] = 0;
          e0->bstate[ABOVE] = BUNDLE_TAIL;
        }
        e0 = next_edge;
      }
    }
    horiz[CLIP] = NH;
    horiz[SUBJ] = NH;

    // Process each edge at this scanbeam boundary
    for (edge = aet; edge; edge = edge->next) {
      exists[CLIP] =
          edge->bundle[ABOVE][CLIP] + (edge->bundle[BELOW][CLIP] << 1);
      exists[SUBJ] =
          edge->bundle[ABOVE][SUBJ] + (edge->bundle[BELOW][SUBJ] << 1);
      if (exists[CLIP] || exists[SUBJ]) {
        /* Set bundle side */
        edge->bside[CLIP] = parity[CLIP];
        edge->bside[SUBJ] = parity[SUBJ];
        /* Determine contributing status and quadrant occupancies */
        switch (op) {
          case GPC_DIFF:
          case GPC_INT:
            contributing = (exists[CLIP] && (parity[SUBJ] || horiz[SUBJ])) ||
                           (exists[SUBJ] && (parity[CLIP] || horiz[CLIP])) ||
                           (exists[CLIP] && exists[SUBJ] &&
                            (parity[CLIP] == parity[SUBJ]));
            br = (parity[CLIP]) && (parity[SUBJ]);
            bl = (parity[CLIP] ^ edge->bundle[ABOVE][CLIP]) &&
                 (parity[SUBJ] ^ edge->bundle[ABOVE][SUBJ]);
            tr = (parity[CLIP] ^ (horiz[CLIP] != NH)) &&
                 (parity[SUBJ] ^ (horiz[SUBJ] != NH));
            tl = (parity[CLIP] ^ (horiz[CLIP] != NH) ^
                  edge->bundle[BELOW][CLIP]) &&
                 (parity[SUBJ] ^ (horiz[SUBJ] != NH) ^
                  edge->bundle[BELOW][SUBJ]);
            break;
          case GPC_XOR:
            contributing = exists[CLIP] || exists[SUBJ];
            br = (parity[CLIP]) ^ (parity[SUBJ]);
            bl = (parity[CLIP] ^ edge->bundle[ABOVE][CLIP]) ^
                 (parity[SUBJ] ^ edge->bundle[ABOVE][SUBJ]);
            tr = (parity[CLIP] ^ (horiz[CLIP] != NH)) ^
                 (parity[SUBJ] ^ (horiz[SUBJ] != NH));
            tl = (parity[CLIP] ^ (horiz[CLIP] != NH) ^
                  edge->bundle[BELOW][CLIP]) ^
                 (parity[SUBJ] ^ (horiz[SUBJ] != NH) ^
                  edge->bundle[BELOW][SUBJ]);
            break;
          case GPC_UNION:
            contributing = (exists[CLIP] && (!parity[SUBJ] || horiz[SUBJ])) ||
                           (exists[SUBJ] && (!parity[CLIP] || horiz[CLIP])) ||
                           (exists[CLIP] && exists[SUBJ] &&
                            (parity[CLIP] == parity[SUBJ]));
            br = (parity[CLIP]) || (parity[SUBJ]);
            bl = (parity[CLIP] ^ edge->bundle[ABOVE][CLIP]) ||
                 (parity[SUBJ] ^ edge->bundle[ABOVE][SUBJ]);
            tr = (parity[CLIP] ^ (horiz[CLIP] != NH)) ||
                 (parity[SUBJ] ^ (horiz[SUBJ] != NH));
            tl = (parity[CLIP] ^ (horiz[CLIP] != NH) ^
                  edge->bundle[BELOW][CLIP]) ||
                 (parity[SUBJ] ^ (horiz[SUBJ] != NH) ^
                  edge->bundle[BELOW][SUBJ]);
            break;
        }
        // Update parity
        parity[CLIP] ^= edge->bundle[ABOVE][CLIP];
        parity[SUBJ] ^= edge->bundle[ABOVE][SUBJ];
        /* Update horizontal state */
        if (exists[CLIP]) {
          horiz[CLIP] = next_h_state[horiz[CLIP]]
                                    [((exists[CLIP] - 1) << 1) + parity[CLIP]];
        }
        if (exists[SUBJ]) {
          horiz[SUBJ] = next_h_state[horiz[SUBJ]]
                                    [((exists[SUBJ] - 1) << 1) + parity[SUBJ]];
        }
        vclass = tr + (tl << 1) + (br << 2) + (bl << 3);
        if (contributing) {
          xb = edge->xb;
          switch (vclass) {
            case EMN:
            case IMN:
              add_local_min(&out_poly, edge, xb, yb);
              px = xb;
              cf = edge->outp[ABOVE];
              break;
            case ERI:
              if (xb != px) {
                add_right(cf, xb, yb);
                px = xb;
              }
              edge->outp[ABOVE] = cf;
              cf = nullptr;
              break;
            case ELI:
              add_left(edge->outp[BELOW], xb, yb);
              px = xb;
              cf = edge->outp[BELOW];
              break;
            case EMX:
              if (xb != px) {
                add_left(cf, xb, yb);
                px = xb;
              }
              merge_right(cf, edge->outp[BELOW], out_poly);
              cf = nullptr;
              break;
            case ILI:
              if (xb != px) {
                add_left(cf, xb, yb);
                px = xb;
              }
              edge->outp[ABOVE] = cf;
              cf = nullptr;
              break;
            case IRI:
              add_right(edge->outp[BELOW], xb, yb);
              px = xb;
              cf = edge->outp[BELOW];
              edge->outp[BELOW] = nullptr;
              break;
            case IMX:
              if (xb != px) {
                add_right(cf, xb, yb);
                px = xb;
              }
              merge_left(cf, edge->outp[BELOW], out_poly);
              cf = nullptr;
              edge->outp[BELOW] = nullptr;
              break;
            case IMM:
              if (xb != px) {
                add_right(cf, xb, yb);
                px = xb;
              }
              merge_left(cf, edge->outp[BELOW], out_poly);
              edge->outp[BELOW] = nullptr;
              add_local_min(&out_poly, edge, xb, yb);
              cf = edge->outp[ABOVE];
              break;
            case EMM:
              if (xb != px) {
                add_left(cf, xb, yb);
                px = xb;
              }
              merge_right(cf, edge->outp[BELOW], out_poly);
              edge->outp[BELOW] = nullptr;
              add_local_min(&out_poly, edge, xb, yb);
              cf = edge->outp[ABOVE];
              break;
            case LED:
              if (edge->bot.y == yb) {
                add_left(edge->outp[BELOW], xb, yb);
              }
              edge->outp[ABOVE] = edge->outp[BELOW];
              px = xb;
              break;
            case RED:
              if (edge->bot.y == yb) {
                add_right(edge->outp[BELOW], xb, yb);
              }
              edge->outp[ABOVE] = edge->outp[BELOW];
              px = xb;
              break;
            default:
              break;
          } /* End of switch */
        }   /* End of contributing conditional */
      }     /* End of edge exists conditional */
    }       // End of AET loop

    /* Delete terminating edges from the AET, otherwise compute xt */
    for (edge = aet; edge; edge = edge->next) {
      if (edge->top.y == yb) {
        prev_edge = edge->prev;
        next_edge = edge->next;
        if (prev_edge) {
          prev_edge->next = next_edge;
        } else {
          aet = next_edge;
        }
        if (next_edge) {
          next_edge->prev = prev_edge;
        }
        /* Copy bundle head state to the adjacent tail edge if required */
        if ((edge->bstate[BELOW] == BUNDLE_HEAD) && prev_edge) {
          if (prev_edge->bstate[BELOW] == BUNDLE_TAIL) {
            prev_edge->outp[BELOW] = edge->outp[BELOW];
            prev_edge->bstate[BELOW] = UNBUNDLED;
            if (prev_edge->prev) {
              if (prev_edge->prev->bstate[BELOW] == BUNDLE_TAIL) {
                prev_edge->bstate[BELOW] = BUNDLE_HEAD;
              }
            }
          }
        }
      } else {
        if (edge->top.y == yt) {
          edge->xt = edge->top.x;
        } else {
          edge->xt = edge->bot.x + edge->dx * (yt - edge->bot.y);
        }
      }
    }

    if (scanbeam < sbt_entries) {
      /* === SCANBEAM INTERIOR PROCESSING ============================== */
      build_intersection_table(&it, aet, dy);
      /* Process each node in the intersection table */
      for (intersect = it; intersect; intersect = intersect->next) {
        e0 = intersect->ie[0];
        e1 = intersect->ie[1];
        /* Only generate output for contributing intersections */
        if ((e0->bundle[ABOVE][CLIP] || e0->bundle[ABOVE][SUBJ]) &&
            (e1->bundle[ABOVE][CLIP] || e1->bundle[ABOVE][SUBJ])) {
          p = e0->outp[ABOVE];
          q = e1->outp[ABOVE];
          ix = intersect->point.x;
          iy = intersect->point.y + yb;

          in[CLIP] = (e0->bundle[ABOVE][CLIP] && !e0->bside[CLIP]) ||
                     (e1->bundle[ABOVE][CLIP] && e1->bside[CLIP]) ||
                     (!e0->bundle[ABOVE][CLIP] && !e1->bundle[ABOVE][CLIP] &&
                      e0->bside[CLIP] && e1->bside[CLIP]);
          in[SUBJ] = (e0->bundle[ABOVE][SUBJ] && !e0->bside[SUBJ]) ||
                     (e1->bundle[ABOVE][SUBJ] && e1->bside[SUBJ]) ||
                     (!e0->bundle[ABOVE][SUBJ] && !e1->bundle[ABOVE][SUBJ] &&
                      e0->bside[SUBJ] && e1->bside[SUBJ]);

          // Determine quadrant occupancies
          switch (op) {
            case GPC_DIFF:
            case GPC_INT:
              tr = (in[CLIP]) && (in[SUBJ]);
              tl = (in[CLIP] ^ e1->bundle[ABOVE][CLIP]) &&
                   (in[SUBJ] ^ e1->bundle[ABOVE][SUBJ]);
              br = (in[CLIP] ^ e0->bundle[ABOVE][CLIP]) &&
                   (in[SUBJ] ^ e0->bundle[ABOVE][SUBJ]);
              bl = (in[CLIP] ^ e1->bundle[ABOVE][CLIP] ^
                    e0->bundle[ABOVE][CLIP]) &&
                   (in[SUBJ] ^ e1->bundle[ABOVE][SUBJ] ^
                    e0->bundle[ABOVE][SUBJ]);
              break;
            case GPC_XOR:
              tr = (in[CLIP]) ^ (in[SUBJ]);
              tl = (in[CLIP] ^ e1->bundle[ABOVE][CLIP]) ^
                   (in[SUBJ] ^ e1->bundle[ABOVE][SUBJ]);
              br = (in[CLIP] ^ e0->bundle[ABOVE][CLIP]) ^
                   (in[SUBJ] ^ e0->bundle[ABOVE][SUBJ]);
              bl = (in[CLIP] ^ e1->bundle[ABOVE][CLIP] ^
                    e0->bundle[ABOVE][CLIP]) ^
                   (in[SUBJ] ^ e1->bundle[ABOVE][SUBJ] ^
                    e0->bundle[ABOVE][SUBJ]);
              break;
            case GPC_UNION:
              tr = (in[CLIP]) || (in[SUBJ]);
              tl = (in[CLIP] ^ e1->bundle[ABOVE][CLIP]) ||
                   (in[SUBJ] ^ e1->bundle[ABOVE][SUBJ]);
              br = (in[CLIP] ^ e0->bundle[ABOVE][CLIP]) ||
                   (in[SUBJ] ^ e0->bundle[ABOVE][SUBJ]);
              bl = (in[CLIP] ^ e1->bundle[ABOVE][CLIP] ^
                    e0->bundle[ABOVE][CLIP]) ||
                   (in[SUBJ] ^ e1->bundle[ABOVE][SUBJ] ^
                    e0->bundle[ABOVE][SUBJ]);
              break;
          }
          vclass = tr + (tl << 1) + (br << 2) + (bl << 3);
          switch (vclass) {
            case EMN:
              add_local_min(&out_poly, e0, ix, iy);
              e1->outp[ABOVE] = e0->outp[ABOVE];
              break;
            case ERI:
              if (p) {
                add_right(p, ix, iy);
                e1->outp[ABOVE] = p;
                e0->outp[ABOVE] = nullptr;
              }
              break;
            case ELI:
              if (q) {
                add_left(q, ix, iy);
                e0->outp[ABOVE] = q;
                e1->outp[ABOVE] = nullptr;
              }
              break;
            case EMX:
              if (p && q) {
                add_left(p, ix, iy);
                merge_right(p, q, out_poly);
                e0->outp[ABOVE] = nullptr;
                e1->outp[ABOVE] = nullptr;
              }
              break;
            case IMN:
              add_local_min(&out_poly, e0, ix, iy);
              e1->outp[ABOVE] = e0->outp[ABOVE];
              break;
            case ILI:
              if (p) {
                add_left(p, ix, iy);
                e1->outp[ABOVE] = p;
                e0->outp[ABOVE] = nullptr;
              }
              break;
            case IRI:
              if (q) {
                add_right(q, ix, iy);
                e0->outp[ABOVE] = q;
                e1->outp[ABOVE] = nullptr;
              }
              break;
            case IMX:
              if (p && q) {
                add_right(p, ix, iy);
                merge_left(p, q, out_poly);
                e0->outp[ABOVE] = nullptr;
                e1->outp[ABOVE] = nullptr;
              }
              break;
            case IMM:
              if (p && q) {
                add_right(p, ix, iy);
                merge_left(p, q, out_poly);
                add_local_min(&out_poly, e0, ix, iy);
                e1->outp[ABOVE] = e0->outp[ABOVE];
              }
              break;
            case EMM:
              if (p && q) {
                add_left(p, ix, iy);
                merge_right(p, q, out_poly);
                add_local_min(&out_poly, e0, ix, iy);
                e1->outp[ABOVE] = e0->outp[ABOVE];
              }
              break;
            default:
              break;
          }  // End of switch
        }    /* End of contributing intersection conditional */

        /* Swap bundle sides in response to edge crossing */
        if (e0->bundle[ABOVE][CLIP]) {
          e1->bside[CLIP] = !e1->bside[CLIP];
        }
        if (e1->bundle[ABOVE][CLIP]) {
          e0->bside[CLIP] = !e0->bside[CLIP];
        }
        if (e0->bundle[ABOVE][SUBJ]) {
          e1->bside[SUBJ] = !e1->bside[SUBJ];
        }
        if (e1->bundle[ABOVE][SUBJ]) {
          e0->bside[SUBJ] = !e0->bside[SUBJ];
        }

        /* Swap e0 and e1 bundles in the AET */
        prev_edge = e0->prev;
        next_edge = e1->next;
        if (next_edge) {
          next_edge->prev = e0;
        }
        if (e0->bstate[ABOVE] == BUNDLE_HEAD) {
          search = 1;
          while (search) {
            prev_edge = prev_edge->prev;
            if (prev_edge) {
              if (prev_edge->bstate[ABOVE] != BUNDLE_TAIL) {
                search = 0;
              }
            } else {
              search = 0;
            }
          }
        }
        if (!prev_edge) {
          aet->prev = e1;
          e1->next = aet;
          aet = e0->next;
        } else {
          prev_edge->next->prev = e1;
          e1->next = prev_edge->next;
          prev_edge->next = e0->next;
        }
        e0->next->prev = prev_edge;
        e1->next->prev = e1;
        e0->next = next_edge;
      } /* End of IT loop*/

      // Prepare for next scanbeam
      for (edge = aet; edge; edge = next_edge) {
        next_edge = edge->next;
        succ_edge = edge->succ;
        if ((edge->top.y == yt) && succ_edge) {
          /* Replace AET edge by its successor */
          succ_edge->outp[BELOW] = edge->outp[ABOVE];
          succ_edge->bstate[BELOW] = edge->bstate[ABOVE];
          succ_edge->bundle[BELOW][CLIP] = edge->bundle[ABOVE][CLIP];
          succ_edge->bundle[BELOW][SUBJ] = edge->bundle[ABOVE][SUBJ];
          prev_edge = edge->prev;
          if (prev_edge) {
            prev_edge->next = succ_edge;
          } else {
            aet = succ_edge;
          }
          if (next_edge) {
            next_edge->prev = succ_edge;
          }
          succ_edge->prev = prev_edge;
          succ_edge->next = next_edge;
        } else {
          /* Update this edge */
          edge->outp[BELOW] = edge->outp[ABOVE];
          edge->bstate[BELOW] = edge->bstate[ABOVE];
          edge->bundle[BELOW][CLIP] = edge->bundle[ABOVE][CLIP];
          edge->bundle[BELOW][SUBJ] = edge->bundle[ABOVE][SUBJ];
          edge->xb = edge->xt;
        }
        edge->outp[ABOVE] = nullptr;
      }
    }
  } /* === END OF SCANBEAM PROCESSING ================================== */
  // Generate result polygon from out_poly
  result->contour = nullptr;
  result->hole = nullptr;
  result->num_contours = count_contours(out_poly);
  if (result->num_contours > 0) {
    gpc_malloc<int>(result->hole,
                    result->num_contours * static_cast<int>(sizeof(int)),
                    const_cast<char *>("hole flag table creation"));  // NOLINT
    gpc_malloc<gpc_vertex_list>(
        result->contour,
        result->num_contours * static_cast<int>(sizeof(gpc_vertex_list)),
        const_cast<char *>("contour creation"));  // NOLINT

    c = 0;
    for (poly = out_poly; poly; poly = npoly) {
      npoly = poly->next;
      if (poly->active) {
        result->hole[c] = poly->proxy->hole;
        result->contour[c].num_vertices = poly->active;
        gpc_malloc<gpc_vertex>(
            result->contour[c].vertex,
            result->contour[c].num_vertices *
                static_cast<int>(sizeof(gpc_vertex)),
            const_cast<char *>("vertex creation"));  // NOLINT

        v = result->contour[c].num_vertices - 1;
        for (vtx = poly->proxy->v[LEFT]; vtx; vtx = nv) {
          nv = vtx->next;
          result->contour[c].vertex[v].x = vtx->x;
          result->contour[c].vertex[v].y = vtx->y;
          gpc_free<vertex_node>(vtx);
          v--;
        }
        c++;
      }
      gpc_free<polygon_node>(poly);
    }
  } else {
    for (poly = out_poly; poly; poly = npoly) {
      npoly = poly->next;
      gpc_free<polygon_node>(poly);
    }
  }

  // Tidy up
  reset_it(&it);
  reset_lmt(&lmt);
  gpc_free<edge_node>(c_heap);
  gpc_free<edge_node>(s_heap);
  gpc_free<double>(sbt);
}  // NOLINT

void gpc_free_tristrip(gpc_tristrip *t) {
  int s = 0;
  for (s = 0; s < t->num_strips; s++) {
    gpc_free<gpc_vertex>(t->strip[s].vertex);
  }
  gpc_free<gpc_vertex_list>(t->strip);
  t->num_strips = 0;
}

void gpc_polygon_to_tristrip(gpc_polygon *s, gpc_tristrip *t) {
  gpc_polygon c;
  c.num_contours = 0;
  c.hole = nullptr;
  c.contour = nullptr;
  gpc_tristrip_clip(GPC_DIFF, s, &c, t);
}

// gpc_tristrip_clip
void gpc_tristrip_clip(gpc_op op,
                       gpc_polygon *subj,
                       gpc_polygon *clip,
                       gpc_tristrip *result) {
  sb_tree *sbtree = nullptr;
  it_node *it = nullptr;
  it_node *intersect = nullptr;
  edge_node *edge = nullptr;
  edge_node *prev_edge = nullptr;
  edge_node *next_edge = nullptr;
  edge_node *succ_edge = nullptr;
  edge_node *e0 = nullptr;
  edge_node *e1 = nullptr;
  edge_node *aet = nullptr;
  edge_node *c_heap = nullptr;
  edge_node *s_heap = nullptr;
  edge_node *cf = nullptr;
  lmt_node *lmt = nullptr;
  lmt_node *local_min = nullptr;
  polygon_node *tlist = nullptr;
  polygon_node *tn = nullptr;
  polygon_node *tnn = nullptr;
  polygon_node *p = nullptr;
  polygon_node *q = nullptr;
  vertex_node *lt = nullptr;
  vertex_node *ltn = nullptr;
  vertex_node *rt = nullptr;
  vertex_node *rtn = nullptr;
  std::array<h_state, 2> horiz = {};
  vertex_type cft = NUL;
  std::array<int, 2> in = {};
  std::array<int, 2> exists = {};
  std::array<int, 2> parity = {LEFT, LEFT};
  int s = 0;
  int v = 0;
  int contributing = 0;
  int search = 0;
  int scanbeam = 0;
  int sbt_entries = 0;
  int vclass = 0;
  int bl = 0;
  int br = 0;
  int tl = 0;
  int tr = 0;
  double *sbt = nullptr;
  double xb = 0.0;
  double px = 0.0;
  double nx = 0.0;
  double yb = 0.0;
  double yt = 0.0;
  double dy = 0.0;
  double ix = 0.0;
  double iy = 0.0;

  /* Test for trivial NULL result cases */
  if (((subj->num_contours == 0) && (clip->num_contours == 0)) ||
      ((subj->num_contours == 0) && ((op == GPC_INT) || (op == GPC_DIFF))) ||
      ((clip->num_contours == 0) && (op == GPC_INT))) {
    result->num_strips = 0;
    result->strip = nullptr;
    return;
  }

  /* Identify potentially contributing contours */
  if (((op == GPC_INT) || (op == GPC_DIFF)) && (subj->num_contours > 0) &&
      (clip->num_contours > 0)) {
    minimax_test(subj, clip, op);
  }
  /* Build LMT */
  if (subj->num_contours > 0) {
    s_heap = build_lmt(&lmt, &sbtree, &sbt_entries, subj, SUBJ, op);
  }
  if (clip->num_contours > 0) {
    c_heap = build_lmt(&lmt, &sbtree, &sbt_entries, clip, CLIP, op);
  }
  /* Return a NULL result if no contours contribute */
  if (lmt == nullptr) {
    result->num_strips = 0;
    result->strip = nullptr;
    reset_lmt(&lmt);
    gpc_free<edge_node>(s_heap);
    gpc_free<edge_node>(c_heap);
    return;
  }

  /* Build scanbeam table from scanbeam tree */
  gpc_malloc<double>(sbt,
                     sbt_entries * static_cast<int>(sizeof(double)),
                     const_cast<char *>("sbt creation"));  // NOLINT
  PADDLE_ENFORCE_NOT_NULL(sbt,
                          common::errors::ResourceExhausted(
                              "Failed to malloc scanbeam table memory."));
  build_sbt(&scanbeam, sbt, sbtree);
  scanbeam = 0;
  free_sbtree(&sbtree);

  /* Invert clip polygon for difference operation */
  if (op == GPC_DIFF) {
    parity[CLIP] = RIGHT;
  }
  local_min = lmt;

  // Process each scanbeam
  while (scanbeam < sbt_entries) {
    /* Set yb and yt to the bottom and top of the scanbeam */
    yb = sbt[scanbeam++];
    if (scanbeam < sbt_entries) {
      yt = sbt[scanbeam];  // NOLINT
      dy = yt - yb;
    }

    /* === SCANBEAM BOUNDARY PROCESSING ================================ */
    /* If LMT node corresponding to yb exists */
    if (local_min) {
      if (local_min->y == yb) {
        /* Add edges starting at this local minimum to the AET */
        for (edge = local_min->first_bound; edge; edge = edge->next_bound) {
          add_edge_to_aet(&aet, edge, nullptr);
        }
        local_min = local_min->next;
      }
    }
    /* Set dummy previous x value */
    /* Create bundles within AET */
    px = -DBL_MAX;
    e0 = aet;
    e1 = aet;  // NOLINT

    /* Set up bundle fields of first edge */
    PADDLE_ENFORCE_NOT_NULL(
        aet, common::errors::InvalidArgument("Edge node AET is nullptr."));
    aet->bundle[ABOVE][aet->type] = (aet->top.y != yb);
    aet->bundle[ABOVE][!aet->type] = 0;
    aet->bstate[ABOVE] = UNBUNDLED;

    for (next_edge = aet->next; next_edge; next_edge = next_edge->next) {
      /* Set up bundle fields of next edge */
      next_edge->bundle[ABOVE][next_edge->type] = (next_edge->top.y != yb);
      next_edge->bundle[ABOVE][!next_edge->type] = 0;
      next_edge->bstate[ABOVE] = UNBUNDLED;

      /* Bundle edges above the scanbeam boundary if they coincide */
      if (next_edge->bundle[ABOVE][next_edge->type]) {
        if (gpc_eq(e0->xb, next_edge->xb) && gpc_eq(e0->dx, next_edge->dx) &&
            (e0->top.y != yb)) {
          next_edge->bundle[ABOVE][next_edge->type] ^=
              e0->bundle[ABOVE][next_edge->type];
          next_edge->bundle[ABOVE][!next_edge->type] =
              e0->bundle[ABOVE][!next_edge->type];
          next_edge->bstate[ABOVE] = BUNDLE_HEAD;
          e0->bundle[ABOVE][CLIP] = 0;
          e0->bundle[ABOVE][SUBJ] = 0;
          e0->bstate[ABOVE] = BUNDLE_TAIL;
        }
        e0 = next_edge;
      }
    }
    horiz[CLIP] = NH;
    horiz[SUBJ] = NH;

    /* Process each edge at this scanbeam boundary */
    for (edge = aet; edge; edge = edge->next) {
      exists[CLIP] =
          edge->bundle[ABOVE][CLIP] + (edge->bundle[BELOW][CLIP] << 1);
      exists[SUBJ] =
          edge->bundle[ABOVE][SUBJ] + (edge->bundle[BELOW][SUBJ] << 1);

      if (exists[CLIP] || exists[SUBJ]) {
        /* Set bundle side */
        edge->bside[CLIP] = parity[CLIP];
        edge->bside[SUBJ] = parity[SUBJ];

        /* Determine contributing status and quadrant occupancies */
        switch (op) {
          case GPC_DIFF:
          case GPC_INT:
            contributing = (exists[CLIP] && (parity[SUBJ] || horiz[SUBJ])) ||
                           (exists[SUBJ] && (parity[CLIP] || horiz[CLIP])) ||
                           (exists[CLIP] && exists[SUBJ] &&
                            (parity[CLIP] == parity[SUBJ]));
            br = (parity[CLIP]) && (parity[SUBJ]);
            bl = (parity[CLIP] ^ edge->bundle[ABOVE][CLIP]) &&
                 (parity[SUBJ] ^ edge->bundle[ABOVE][SUBJ]);
            tr = (parity[CLIP] ^ (horiz[CLIP] != NH)) &&
                 (parity[SUBJ] ^ (horiz[SUBJ] != NH));
            tl = (parity[CLIP] ^ (horiz[CLIP] != NH) ^
                  edge->bundle[BELOW][CLIP]) &&
                 (parity[SUBJ] ^ (horiz[SUBJ] != NH) ^
                  edge->bundle[BELOW][SUBJ]);
            break;
          case GPC_XOR:
            contributing = exists[CLIP] || exists[SUBJ];
            br = (parity[CLIP]) ^ (parity[SUBJ]);
            bl = (parity[CLIP] ^ edge->bundle[ABOVE][CLIP]) ^
                 (parity[SUBJ] ^ edge->bundle[ABOVE][SUBJ]);
            tr = (parity[CLIP] ^ (horiz[CLIP] != NH)) ^
                 (parity[SUBJ] ^ (horiz[SUBJ] != NH));
            tl = (parity[CLIP] ^ (horiz[CLIP] != NH) ^
                  edge->bundle[BELOW][CLIP]) ^
                 (parity[SUBJ] ^ (horiz[SUBJ] != NH) ^
                  edge->bundle[BELOW][SUBJ]);
            break;
          case GPC_UNION:
            contributing = (exists[CLIP] && (!parity[SUBJ] || horiz[SUBJ])) ||
                           (exists[SUBJ] && (!parity[CLIP] || horiz[CLIP])) ||
                           (exists[CLIP] && exists[SUBJ] &&
                            (parity[CLIP] == parity[SUBJ]));
            br = (parity[CLIP]) || (parity[SUBJ]);
            bl = (parity[CLIP] ^ edge->bundle[ABOVE][CLIP]) ||
                 (parity[SUBJ] ^ edge->bundle[ABOVE][SUBJ]);
            tr = (parity[CLIP] ^ (horiz[CLIP] != NH)) ||
                 (parity[SUBJ] ^ (horiz[SUBJ] != NH));
            tl = (parity[CLIP] ^ (horiz[CLIP] != NH) ^
                  edge->bundle[BELOW][CLIP]) ||
                 (parity[SUBJ] ^ (horiz[SUBJ] != NH) ^
                  edge->bundle[BELOW][SUBJ]);
            break;
        }

        // Update parity
        parity[CLIP] ^= edge->bundle[ABOVE][CLIP];
        parity[SUBJ] ^= edge->bundle[ABOVE][SUBJ];

        /* Update horizontal state */
        if (exists[CLIP]) {
          horiz[CLIP] = next_h_state[horiz[CLIP]]
                                    [((exists[CLIP] - 1) << 1) + parity[CLIP]];
        }
        if (exists[SUBJ]) {
          horiz[SUBJ] = next_h_state[horiz[SUBJ]]
                                    [((exists[SUBJ] - 1) << 1) + parity[SUBJ]];
        }
        vclass = tr + (tl << 1) + (br << 2) + (bl << 3);

        if (contributing) {
          xb = edge->xb;
          switch (vclass) {
            case EMN:
              new_tristrip(&tlist, edge, xb, yb);
              cf = edge;
              break;
            case ERI:
              edge->outp[ABOVE] = cf->outp[ABOVE];
              if (xb != cf->xb) {
                gpc_vertex_create(edge, ABOVE, RIGHT, xb, yb);
              }
              cf = nullptr;
              break;
            case ELI:
              gpc_vertex_create(edge, BELOW, LEFT, xb, yb);
              edge->outp[ABOVE] = nullptr;
              cf = edge;
              break;
            case EMX:
              if (xb != cf->xb) {
                gpc_vertex_create(edge, BELOW, RIGHT, xb, yb);
              }
              edge->outp[ABOVE] = nullptr;
              cf = nullptr;
              break;
            case IMN:
              if (cft == LED) {
                if (cf->bot.y != yb) {
                  gpc_vertex_create(cf, BELOW, LEFT, cf->xb, yb);
                }
                new_tristrip(&tlist, cf, cf->xb, yb);
              }
              if (cf) edge->outp[ABOVE] = cf->outp[ABOVE];
              gpc_vertex_create(edge, ABOVE, RIGHT, xb, yb);
              break;
            case ILI:
              new_tristrip(&tlist, edge, xb, yb);
              cf = edge;
              cft = ILI;
              break;
            case IRI:
              if (cft == LED) {
                if (cf->bot.y != yb) {
                  gpc_vertex_create(cf, BELOW, LEFT, cf->xb, yb);
                }
                new_tristrip(&tlist, cf, cf->xb, yb);
              }
              gpc_vertex_create(edge, BELOW, RIGHT, xb, yb);
              edge->outp[ABOVE] = nullptr;
              break;
            case IMX:
              gpc_vertex_create(edge, BELOW, LEFT, xb, yb);
              edge->outp[ABOVE] = nullptr;
              cft = IMX;
              break;
            case IMM:
              gpc_vertex_create(edge, BELOW, LEFT, xb, yb);
              edge->outp[ABOVE] = cf->outp[ABOVE];
              if (xb != cf->xb) {
                gpc_vertex_create(cf, ABOVE, RIGHT, xb, yb);
              }
              cf = edge;
              break;
            case EMM:
              gpc_vertex_create(edge, BELOW, RIGHT, xb, yb);
              edge->outp[ABOVE] = nullptr;
              new_tristrip(&tlist, edge, xb, yb);
              cf = edge;
              break;
            case LED:
              if (edge->bot.y == yb) {
                gpc_vertex_create(edge, BELOW, LEFT, xb, yb);
              }
              edge->outp[ABOVE] = edge->outp[BELOW];
              cf = edge;
              cft = LED;
              break;
            case RED:
              edge->outp[ABOVE] = cf->outp[ABOVE];
              if (cft == LED) {
                if (cf->bot.y == yb) {
                  gpc_vertex_create(edge, BELOW, RIGHT, xb, yb);
                } else {
                  if (edge->bot.y == yb) {
                    gpc_vertex_create(cf, BELOW, LEFT, cf->xb, yb);
                    gpc_vertex_create(edge, BELOW, RIGHT, xb, yb);
                  }
                }
              } else {
                gpc_vertex_create(edge, BELOW, RIGHT, xb, yb);
                gpc_vertex_create(edge, ABOVE, RIGHT, xb, yb);
              }
              cf = nullptr;
              break;
            default:
              break;
          } /* End of switch */
        }   /* End of contributing conditional */
      }     /* End of edge exists conditional */
    }       // End of AET loop

    /* Delete terminating edges from the AET, otherwise compute xt */
    for (edge = aet; edge; edge = edge->next) {
      if (edge->top.y == yb) {
        prev_edge = edge->prev;
        next_edge = edge->next;
        if (prev_edge) {
          prev_edge->next = next_edge;
        } else {
          aet = next_edge;
        }
        if (next_edge) {
          next_edge->prev = prev_edge;
        }

        /* Copy bundle head state to the adjacent tail edge if required */
        if ((edge->bstate[BELOW] == BUNDLE_HEAD) && prev_edge) {
          if (prev_edge->bstate[BELOW] == BUNDLE_TAIL) {
            prev_edge->outp[BELOW] = edge->outp[BELOW];
            prev_edge->bstate[BELOW] = UNBUNDLED;
            if (prev_edge->prev) {
              if (prev_edge->prev->bstate[BELOW] == BUNDLE_TAIL) {
                prev_edge->bstate[BELOW] = BUNDLE_HEAD;
              }
            }
          }
        }
      } else {
        if (edge->top.y == yt) {
          edge->xt = edge->top.x;
        } else {
          edge->xt = edge->bot.x + edge->dx * (yt - edge->bot.y);
        }
      }
    }

    if (scanbeam < sbt_entries) {
      /* === SCANBEAM INTERIOR PROCESSING ============================== */
      build_intersection_table(&it, aet, dy);
      /* Process each node in the intersection table */
      for (intersect = it; intersect; intersect = intersect->next) {
        e0 = intersect->ie[0];
        e1 = intersect->ie[1];

        /* Only generate output for contributing intersections */
        if ((e0->bundle[ABOVE][CLIP] || e0->bundle[ABOVE][SUBJ]) &&
            (e1->bundle[ABOVE][CLIP] || e1->bundle[ABOVE][SUBJ])) {
          p = e0->outp[ABOVE];
          q = e1->outp[ABOVE];
          ix = intersect->point.x;
          iy = intersect->point.y + yb;

          in[CLIP] = (e0->bundle[ABOVE][CLIP] && !e0->bside[CLIP]) ||
                     (e1->bundle[ABOVE][CLIP] && e1->bside[CLIP]) ||
                     (!e0->bundle[ABOVE][CLIP] && !e1->bundle[ABOVE][CLIP] &&
                      e0->bside[CLIP] && e1->bside[CLIP]);
          in[SUBJ] = (e0->bundle[ABOVE][SUBJ] && !e0->bside[SUBJ]) ||
                     (e1->bundle[ABOVE][SUBJ] && e1->bside[SUBJ]) ||
                     (!e0->bundle[ABOVE][SUBJ] && !e1->bundle[ABOVE][SUBJ] &&
                      e0->bside[SUBJ] && e1->bside[SUBJ]);

          switch (op) {  // Determine quadrant occupancies
            case GPC_DIFF:
            case GPC_INT:
              tr = (in[CLIP]) && (in[SUBJ]);
              tl = (in[CLIP] ^ e1->bundle[ABOVE][CLIP]) &&
                   (in[SUBJ] ^ e1->bundle[ABOVE][SUBJ]);
              br = (in[CLIP] ^ e0->bundle[ABOVE][CLIP]) &&
                   (in[SUBJ] ^ e0->bundle[ABOVE][SUBJ]);
              bl = (in[CLIP] ^ e1->bundle[ABOVE][CLIP] ^
                    e0->bundle[ABOVE][CLIP]) &&
                   (in[SUBJ] ^ e1->bundle[ABOVE][SUBJ] ^
                    e0->bundle[ABOVE][SUBJ]);
              break;
            case GPC_XOR:
              tr = (in[CLIP]) ^ (in[SUBJ]);
              tl = (in[CLIP] ^ e1->bundle[ABOVE][CLIP]) ^
                   (in[SUBJ] ^ e1->bundle[ABOVE][SUBJ]);
              br = (in[CLIP] ^ e0->bundle[ABOVE][CLIP]) ^
                   (in[SUBJ] ^ e0->bundle[ABOVE][SUBJ]);
              bl = (in[CLIP] ^ e1->bundle[ABOVE][CLIP] ^
                    e0->bundle[ABOVE][CLIP]) ^
                   (in[SUBJ] ^ e1->bundle[ABOVE][SUBJ] ^
                    e0->bundle[ABOVE][SUBJ]);
              break;
            case GPC_UNION:
              tr = (in[CLIP]) || (in[SUBJ]);
              tl = (in[CLIP] ^ e1->bundle[ABOVE][CLIP]) ||
                   (in[SUBJ] ^ e1->bundle[ABOVE][SUBJ]);
              br = (in[CLIP] ^ e0->bundle[ABOVE][CLIP]) ||
                   (in[SUBJ] ^ e0->bundle[ABOVE][SUBJ]);
              bl = (in[CLIP] ^ e1->bundle[ABOVE][CLIP] ^
                    e0->bundle[ABOVE][CLIP]) ||
                   (in[SUBJ] ^ e1->bundle[ABOVE][SUBJ] ^
                    e0->bundle[ABOVE][SUBJ]);
              break;
          }

          vclass = tr + (tl << 1) + (br << 2) + (bl << 3);
          switch (vclass) {
            case EMN:
              new_tristrip(&tlist, e1, ix, iy);
              e0->outp[ABOVE] = e1->outp[ABOVE];
              break;
            case ERI:
              if (p) {
                gpc_p_edge(prev_edge, e0, ABOVE);
                gpc_vertex_create(prev_edge, ABOVE, LEFT, px, iy);
                gpc_vertex_create(e0, ABOVE, RIGHT, ix, iy);
                e1->outp[ABOVE] = e0->outp[ABOVE];
                e0->outp[ABOVE] = nullptr;
              }
              break;
            case ELI:
              if (q) {
                gpc_n_edge(next_edge, e1, ABOVE);
                gpc_vertex_create(e1, ABOVE, LEFT, ix, iy);
                gpc_vertex_create(next_edge, ABOVE, RIGHT, nx, iy);
                e0->outp[ABOVE] = e1->outp[ABOVE];
                e1->outp[ABOVE] = nullptr;
              }
              break;
            case EMX:
              if (p && q) {
                gpc_vertex_create(e0, ABOVE, LEFT, ix, iy);
                e0->outp[ABOVE] = nullptr;
                e1->outp[ABOVE] = nullptr;
              }
              break;
            case IMN:
              gpc_p_edge(prev_edge, e0, ABOVE);
              gpc_vertex_create(prev_edge, ABOVE, LEFT, px, iy);
              gpc_n_edge(next_edge, e1, ABOVE);
              gpc_vertex_create(next_edge, ABOVE, RIGHT, nx, iy);
              new_tristrip(&tlist, prev_edge, px, iy);
              e1->outp[ABOVE] = prev_edge->outp[ABOVE];
              gpc_vertex_create(e1, ABOVE, RIGHT, ix, iy);
              new_tristrip(&tlist, e0, ix, iy);
              next_edge->outp[ABOVE] = e0->outp[ABOVE];
              gpc_vertex_create(next_edge, ABOVE, RIGHT, nx, iy);
              break;
            case ILI:
              if (p) {
                gpc_vertex_create(e0, ABOVE, LEFT, ix, iy);
                gpc_n_edge(next_edge, e1, ABOVE);
                gpc_vertex_create(next_edge, ABOVE, RIGHT, nx, iy);
                e1->outp[ABOVE] = e0->outp[ABOVE];
                e0->outp[ABOVE] = nullptr;
              }
              break;
            case IRI:
              if (q) {
                gpc_vertex_create(e1, ABOVE, RIGHT, ix, iy);
                gpc_p_edge(prev_edge, e0, ABOVE);
                gpc_vertex_create(prev_edge, ABOVE, LEFT, px, iy);
                e0->outp[ABOVE] = e1->outp[ABOVE];
                e1->outp[ABOVE] = nullptr;
              }
              break;
            case IMX:
              if (p && q) {
                gpc_vertex_create(e0, ABOVE, RIGHT, ix, iy);
                gpc_vertex_create(e1, ABOVE, LEFT, ix, iy);
                e0->outp[ABOVE] = nullptr;
                e1->outp[ABOVE] = nullptr;
                gpc_p_edge(prev_edge, e0, ABOVE);
                gpc_vertex_create(prev_edge, ABOVE, LEFT, px, iy);
                new_tristrip(&tlist, prev_edge, px, iy);
                gpc_n_edge(next_edge, e1, ABOVE);
                gpc_vertex_create(next_edge, ABOVE, RIGHT, nx, iy);
                next_edge->outp[ABOVE] = prev_edge->outp[ABOVE];
                gpc_vertex_create(next_edge, ABOVE, RIGHT, nx, iy);
              }
              break;
            case IMM:
              if (p && q) {
                gpc_vertex_create(e0, ABOVE, RIGHT, ix, iy);
                gpc_vertex_create(e1, ABOVE, LEFT, ix, iy);
                gpc_p_edge(prev_edge, e0, ABOVE);
                gpc_vertex_create(prev_edge, ABOVE, LEFT, px, iy);
                new_tristrip(&tlist, prev_edge, px, iy);
                gpc_n_edge(next_edge, e1, ABOVE);
                gpc_vertex_create(next_edge, ABOVE, RIGHT, nx, iy);
                e1->outp[ABOVE] = prev_edge->outp[ABOVE];
                gpc_vertex_create(e1, ABOVE, RIGHT, ix, iy);
                new_tristrip(&tlist, e0, ix, iy);
                next_edge->outp[ABOVE] = e0->outp[ABOVE];
                gpc_vertex_create(next_edge, ABOVE, RIGHT, nx, iy);
              }
              break;
            case EMM:
              if (p && q) {
                gpc_vertex_create(e0, ABOVE, LEFT, ix, iy);
                new_tristrip(&tlist, e1, ix, iy);
                e0->outp[ABOVE] = e1->outp[ABOVE];
              }
              break;
            default:
              break;
          } /* End of switch */
        }   /* End of contributing intersection conditional */

        // Swap bundle sides in response to edge crossing
        if (e0->bundle[ABOVE][CLIP]) {
          e1->bside[CLIP] = !e1->bside[CLIP];
        }
        if (e1->bundle[ABOVE][CLIP]) {
          e0->bside[CLIP] = !e0->bside[CLIP];
        }
        if (e0->bundle[ABOVE][SUBJ]) {
          e1->bside[SUBJ] = !e1->bside[SUBJ];
        }
        if (e1->bundle[ABOVE][SUBJ]) {
          e0->bside[SUBJ] = !e0->bside[SUBJ];
        }

        /* Swap e0 and e1 bundles in the AET */
        prev_edge = e0->prev;
        next_edge = e1->next;
        if (e1->next) {
          e1->next->prev = e0;
        }

        if (e0->bstate[ABOVE] == BUNDLE_HEAD) {
          search = 1;
          while (search) {
            prev_edge = prev_edge->prev;
            if (prev_edge) {
              if (prev_edge->bundle[ABOVE][CLIP] ||
                  prev_edge->bundle[ABOVE][SUBJ] ||
                  (prev_edge->bstate[ABOVE] == BUNDLE_HEAD)) {
                search = 0;
              }
            } else {
              search = 0;
            }
          }
        }
        if (!prev_edge) {
          e1->next = aet;
          aet = e0->next;
        } else {
          e1->next = prev_edge->next;
          prev_edge->next = e0->next;
        }
        e0->next->prev = prev_edge;
        e1->next->prev = e1;
        e0->next = next_edge;
      } /* End of IT loop*/

      /* Prepare for next scanbeam */
      for (edge = aet; edge; edge = next_edge) {
        next_edge = edge->next;
        succ_edge = edge->succ;

        if ((edge->top.y == yt) && succ_edge) {
          /* Replace AET edge by its successor */
          succ_edge->outp[BELOW] = edge->outp[ABOVE];
          succ_edge->bstate[BELOW] = edge->bstate[ABOVE];
          succ_edge->bundle[BELOW][CLIP] = edge->bundle[ABOVE][CLIP];
          succ_edge->bundle[BELOW][SUBJ] = edge->bundle[ABOVE][SUBJ];
          prev_edge = edge->prev;
          if (prev_edge) {
            prev_edge->next = succ_edge;
          } else {
            aet = succ_edge;
          }
          if (next_edge) {
            next_edge->prev = succ_edge;
          }
          succ_edge->prev = prev_edge;
          succ_edge->next = next_edge;
        } else {
          /* Update this edge */
          edge->outp[BELOW] = edge->outp[ABOVE];
          edge->bstate[BELOW] = edge->bstate[ABOVE];
          edge->bundle[BELOW][CLIP] = edge->bundle[ABOVE][CLIP];
          edge->bundle[BELOW][SUBJ] = edge->bundle[ABOVE][SUBJ];
          edge->xb = edge->xt;
        }
        edge->outp[ABOVE] = nullptr;
      }
    }
  } /* === END OF SCANBEAM PROCESSING ================================== */

  // Generate result tristrip from tlist
  result->strip = nullptr;
  result->num_strips = count_tristrips(tlist);
  if (result->num_strips > 0) {
    gpc_malloc<gpc_vertex_list>(
        result->strip,
        result->num_strips * static_cast<int>(sizeof(gpc_vertex_list)),
        const_cast<char *>("tristrip list creation"));  // NOLINT

    s = 0;
    for (tn = tlist; tn; tn = tnn) {
      tnn = tn->next;
      if (tn->active > 2) {
        /* Valid tristrip: copy the vertices and free the heap */
        result->strip[s].num_vertices = tn->active;
        gpc_malloc<gpc_vertex>(
            result->strip[s].vertex,
            tn->active * static_cast<int>(sizeof(gpc_vertex)),
            const_cast<char *>("tristrip creation"));  // NOLINT
        v = 0;
        if (false) {
          lt = tn->v[RIGHT];
          rt = tn->v[LEFT];
        } else {
          lt = tn->v[LEFT];
          rt = tn->v[RIGHT];
        }
        while (lt || rt) {
          if (lt) {
            ltn = lt->next;
            result->strip[s].vertex[v].x = lt->x;
            result->strip[s].vertex[v].y = lt->y;
            v++;
            gpc_free<vertex_node>(lt);
            lt = ltn;
          }
          if (rt) {
            rtn = rt->next;
            result->strip[s].vertex[v].x = rt->x;
            result->strip[s].vertex[v].y = rt->y;
            v++;
            gpc_free<vertex_node>(rt);
            rt = rtn;
          }
        }
        s++;
      } else {
        /* Invalid tristrip: just free the heap */
        for (lt = tn->v[LEFT]; lt; lt = ltn) {
          ltn = lt->next;
          gpc_free<vertex_node>(lt);
        }
        for (rt = tn->v[RIGHT]; rt; rt = rtn) {
          rtn = rt->next;
          gpc_free<vertex_node>(rt);
        }
      }
      gpc_free<polygon_node>(tn);
    }
  }
  // Tidy up
  reset_it(&it);
  reset_lmt(&lmt);
  gpc_free<edge_node>(c_heap);
  gpc_free<edge_node>(s_heap);
  gpc_free<double>(sbt);
}  // NOLINT

}  // namespace phi::funcs
