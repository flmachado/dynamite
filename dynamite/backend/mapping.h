
#pragma once
#include <petsc.h>

#define IDX(L,n,k) ((n)+(k)*(L))

typedef struct _map_ctx
{
  PetscInt L;
  PetscInt sz;
  PetscInt start;
  PetscInt* map;
  PetscInt* choose;
} map_ctx;

PetscErrorCode BuildMapCtx(PetscInt L, PetscInt sz, map_ctx **c_p);
PetscErrorCode BuildMapArray(PetscInt start,PetscInt end,map_ctx *c);
PetscErrorCode FreeMapCtx(map_ctx *c);

static inline PetscInt MaxIdx(map_ctx *c) {
  if (c->sz == 0) return 1;
  else return c->choose[IDX(c->L,c->L-1,c->sz)] + c->choose[IDX(c->L,c->L-1,c->sz-1)];
}

static inline PetscInt MapForwardSingle(map_ctx* c,PetscInt idx) {
  return c->map[idx - c->start];
}

static inline void MapForward(map_ctx* c,PetscInt n,PetscInt *idxs,PetscInt *states) {
  PetscInt i;
  for (i=0;i<n;++i) {
    states[i] = c->map[idxs[i] - c->start];
  }
}

static inline PetscInt MapReverseSingle(map_ctx *c,PetscInt state) {
  PetscInt n,k,idx = 0;

  /* if state is outside of the subspace, return -1 */
  if (__builtin_popcount(state) != c->sz) return -1;

  k = 0;
  while (state != 0) {
    ++k;
    n = __builtin_ctz(state);
    idx += c->choose[IDX(c->L,n,k)];

    /* zero out the lowest 1 bit */
    state = state & (state-1);
  }

  return idx;
}

static inline void MapReverse(map_ctx *c,PetscInt n,PetscInt* states,PetscInt* idxs) {
  int i;
  for (i=0;i<n;++i) {
    idxs[i] = MapReverseSingle(c,states[i]);
  }
}
