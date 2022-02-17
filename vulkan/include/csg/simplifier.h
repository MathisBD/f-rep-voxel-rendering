#pragma once
#include "csg/expression.h"


namespace csg
{
    // Merge all the X, Y, Z and T nodes
    // so that there is only one left of each type.
    Expr MergeAxes(Expr e);
    // Carry out all operations whose inputs are constants
    // and replace some operations using simple algebraic identities.
    Expr ConstantFold(Expr e);

    int ThreeWayCompare(Expr a, Expr b);
    Expr NormalForm(Expr e);
    Expr MergeDuplicates(Expr e);
}