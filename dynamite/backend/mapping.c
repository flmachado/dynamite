
#include "mapping.h"

PetscErrorCode BuildMapCtx(PetscInt L, PetscInt sz, map_ctx **c_p) {
  PetscInt i,j;
  PetscErrorCode ierr;

  ierr = PetscMalloc1(1,c_p);CHKERRQ(ierr);

  (*c_p)->L = L;
  (*c_p)->sz = sz;
  (*c_p)->map = NULL;
  (*c_p)->start = -1;

  ierr = PetscMalloc1(L*(sz + 1),&((*c_p)->choose));CHKERRQ(ierr);

  /* compute the values of i choose j */
  for (j=1;j<sz+1;++j) {
    (*c_p)->choose[IDX(L,0,j)] = 0;
  }

  for (i=0;i<L;++i) (*c_p)->choose[IDX(L,i,0)] = 1;

  for (i=1;i<L;++i) {
    for (j=1;j<sz+1;++j) {
      (*c_p)->choose[IDX(L,i,j)] = (*c_p)->choose[IDX(L,i-1,j)] + \
                                   (*c_p)->choose[IDX(L,i-1,j-1)];
    }
  }

  return ierr;
}

PetscErrorCode BuildMapArray(PetscInt start,PetscInt end,map_ctx *c) {
  PetscInt i,i_tmp,n,k;
  PetscInt *s;
  PetscErrorCode ierr;

  if (c->map != NULL) {
    ierr = PetscFree(c->map);CHKERRQ(ierr);
  }

  c->start = start;

  ierr = PetscMalloc1(end-start,&(c->map));CHKERRQ(ierr);

  for (i=start;i<end;++i) {
    n = c->L;
    k = c->sz;
    s = c->map + (i-start);
    i_tmp = i;
    (*s) = 0;

    while(k>0) {
      if (i_tmp >= c->choose[IDX(c->L,n-1,k)]) {
        i_tmp -= c->choose[IDX(c->L,n-1,k)];
        --k;
        /* set a 1 bit in the correct spot */
        (*s) |= (1<<(n-1));
      }
      --n;
    }
  }

  return ierr;
}

PetscErrorCode FreeMapCtx(map_ctx *c) {
  PetscErrorCode ierr;
  if (c->map != NULL) {
    ierr = PetscFree(c->map);CHKERRQ(ierr);
  }
  ierr = PetscFree(c->choose);CHKERRQ(ierr);
  ierr = PetscFree(c);CHKERRQ(ierr);
  return ierr;
}
