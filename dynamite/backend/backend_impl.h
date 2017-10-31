#ifndef BACKEND_H
#define BACKEND_H

#include <slepcmfn.h>
#include "shellcontext.h"
#include "mapping.h"

PetscErrorCode BuildMat_Full(PetscInt L,PetscInt nterms,PetscInt sz,PetscInt* masks,PetscInt* signs,PetscScalar* coeffs,Mat *A);
PetscErrorCode BuildMat_Shell(PetscInt L,PetscInt nterms,PetscInt* masks,PetscInt* signs,PetscScalar* coeffs,Mat *A);
PetscErrorCode MatMult_Shell(Mat A,Vec x,Vec b);
PetscErrorCode MatNorm_Shell(Mat A,NormType type,PetscReal *nrm);
PetscErrorCode BuildContext(PetscInt L,PetscInt nterms,PetscInt* masks,PetscInt* signs,PetscScalar* coeffs,shell_context **ctx_p);
PetscErrorCode DestroyContext(Mat A);

PetscErrorCode ReducedDensityMatrix(PetscInt L,Vec x,PetscInt cut_size,PetscInt start,PetscBool fillall,PetscScalar* m);
PetscErrorCode ReducedDensityMatrix_SC(PetscInt L,PetscInt sz,Vec x,PetscInt cut_size,PetscInt start,PetscBool fillall,PetscScalar* m);

#endif /* !BACKEND_H */
