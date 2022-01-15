#include "csg/simplifier.h"
#include <algorithm>

csg::Expr csg::MergeAxes(csg::Expr root)
{
    csg::Expr x, y, z, t;
    return root.TopoMap<csg::Expr>([&] (csg::Expr e, std::vector<csg::Expr> inputs) {
        switch (e.node->op) {
        case csg::Operator::X: 
            if (x.node.get() == nullptr) { x = e; }
            return x;
        case csg::Operator::Y: 
            if (y.node.get() == nullptr) { y = e; }
            return y;
        case csg::Operator::Z: 
            if (z.node.get() == nullptr) { z = e; }
            return z;
        case csg::Operator::T: 
            if (t.node.get() == nullptr) { t = e; }
            return t;
        case csg::Operator::CONST:
            return e;
        default: 
            return csg::Expr(std::make_shared<csg::Node>(e.node->op, std::move(inputs)));
        }
    });
}


// Perform a single constant fold step (i.e. don't recurse on the inputs)
static csg::Expr ConstantFoldStep(csg::Expr e) 
{    
    if (!e.IsInputOp()) { return e; }
    auto inputs = e.node->inputs;
    
    std::vector<float> constants;
    for (auto i : e.node->inputs) {
        if (i.IsConstantOp()) {
            constants.push_back(i.node->constant);
        }
    }
    // All inputs are constant
    if (constants.size() == inputs.size()) {
        float res = csg::ApplyOperator(e.node->op, constants);
        return csg::Expr(res);
    }

    // Some inputs are constant :
    // we can replace some operations with simpler ones.
    switch (e.node->op) {
    case csg::Operator::ADD:
        assert(inputs.size() == 2);
        if (inputs[0].IsConstantOp(0)) { return inputs[1]; }
        if (inputs[1].IsConstantOp(0)) { return inputs[0]; }
        if (inputs[0].IsOp(csg::Operator::NEG) && inputs[1].IsOp(csg::Operator::NEG)) { 
            return -(inputs[0] + inputs[1]); 
        }
        if (inputs[0].IsOp(csg::Operator::NEG)) { return inputs[1] - inputs[0][0]; }
        if (inputs[1].IsOp(csg::Operator::NEG)) { return inputs[0] - inputs[1][0]; }
        break;
    case csg::Operator::MUL:
        assert(inputs.size() == 2);
        if (inputs[0].IsConstantOp(0)) { return csg::Expr(0); }
        if (inputs[1].IsConstantOp(0)) { return csg::Expr(0); }
        if (inputs[0].IsConstantOp(1)) { return inputs[1]; }
        if (inputs[1].IsConstantOp(1)) { return inputs[0]; }
        if (inputs[0].IsConstantOp(-1)) { return ConstantFoldStep(-inputs[1]); }
        if (inputs[1].IsConstantOp(-1)) { return ConstantFoldStep(-inputs[0]); }
        if (inputs[0].IsOp(csg::Operator::NEG) || inputs[1].IsOp(csg::Operator::NEG)) {
            return ConstantFoldStep(-inputs[0]) * ConstantFoldStep(-inputs[1]);
        }
        break;
    case csg::Operator::SUB:
        assert(inputs.size() == 2);
        if (inputs[0].IsConstantOp(0)) { return ConstantFoldStep(-inputs[1]); }
        if (inputs[1].IsConstantOp(0)) { return inputs[0]; }
        if (inputs[1].IsOp(csg::Operator::NEG)) { return ConstantFoldStep(inputs[0] + inputs[1][0]); }
        break;
    case csg::Operator::DIV:
        assert(inputs.size() == 2);
        if (inputs[0].IsConstantOp(0))        { return csg::Expr(0); }
        if (inputs[1].IsConstantOp(1))        { return inputs[0]; }
        if (inputs[1].IsConstantOp(-1))        { return ConstantFoldStep(-inputs[0]); }
        if (e.node->inputs[1].IsConstantOp()) { return ConstantFoldStep(inputs[0] * (1.0f / inputs[1].node->constant)); }
        break;
    case csg::Operator::NEG:
        assert(inputs.size() == 1);
        if (inputs[0].IsOp(csg::Operator::NEG)) { return inputs[0][0]; }
        if (inputs[0].IsOp(csg::Operator::SUB)) { return ConstantFoldStep(inputs[0][1] - inputs[0][0]); }
        break;
    default: break;
    }

    return e;   
}

csg::Expr csg::ConstantFold(csg::Expr root) 
{
    return root.TopoMap<csg::Expr>([=] (csg::Expr e, std::vector<csg::Expr> inputs) {
        if (!e.IsInputOp()) { return e; }
        csg::Expr newE(std::make_shared<csg::Node>(e.node->op, std::move(inputs)));
        return ConstantFoldStep(newE);
    });
}


static constexpr int OpOrder(csg::Operator op)
{
    switch (op) {
    case csg::Operator::X:     return 0;
    case csg::Operator::Y:     return 1;
    case csg::Operator::Z:     return 2;
    case csg::Operator::T:     return 3;
    case csg::Operator::CONST: return 4;
    case csg::Operator::SIN:   return 5;
    case csg::Operator::COS:   return 6;
    case csg::Operator::EXP:   return 7;
    case csg::Operator::NEG:   return 8;
    case csg::Operator::SQRT:  return 9;
    case csg::Operator::ADD:   return 10;
    case csg::Operator::SUB:   return 11;
    case csg::Operator::MUL:   return 12;
    case csg::Operator::DIV:   return 13;
    case csg::Operator::MIN:   return 14;
    case csg::Operator::MAX:   return 15;
    default: assert(false); return -1;
    }
}

int csg::ThreeWayCompare(csg::Expr a, csg::Expr b) 
{
    if (OpOrder(a.node->op) < OpOrder(b.node->op)) {
        return -1;
    }
    else if (OpOrder(a.node->op) > OpOrder(b.node->op)) {
        return 1;
    }

    // Constant op
    auto op = a.node->op;
    if (op == csg::Operator::CONST) {
        if (a.node->constant < b.node->constant) {
            return -1;
        }
        else if (a.node->constant > b.node->constant) {
            return 1;
        }
        else {
            return 0;
        }
    }

    // Check a and b have the same input number
    if (a.node->inputs.size() < b.node->inputs.size()) {
        return -1;
    }
    else if (a.node->inputs.size() > b.node->inputs.size()) {
        return 1;
    }

    // Compare the inputs
    for (size_t i = 0; i < a.node->inputs.size(); i++) {
        int cmp = ThreeWayCompare(a.node->inputs[i], b.node->inputs[i]);
        if (cmp != 0) {
            return cmp;
        }
    }
    return 0;
}


csg::Expr NormalFormStep(csg::Expr e)
{
    switch (e.node->op) {
    // These are commutative operations :
    // we have to sort their operands
    case csg::Operator::ADD:
    case csg::Operator::MUL:
    case csg::Operator::MIN:
    case csg::Operator::MAX: 
        {
            auto inputs = e.node->inputs;
            std::sort(inputs.begin(), inputs.end(), 
                [] (const csg::Expr a, const csg::Expr b) 
            {
                return ThreeWayCompare(a, b) <= 0;
            });
            return std::make_shared<csg::Node>(e.node->op, std::move(inputs));
        }
    default: return e;
    }
}

csg::Expr csg::NormalForm(csg::Expr root) 
{
    return root.TopoMap<csg::Expr>([=] (csg::Expr e, std::vector<csg::Expr> inputs) {
        if (!e.IsInputOp()) { return e; }
        csg::Expr newE(std::make_shared<csg::Node>(e.node->op, std::move(inputs)));
        return NormalFormStep(newE);
    });
}


// Returns true if a and b refer to semantically equivalent expressions.
// Assumes :
// 1. a and b are in normal form
// 2. we have merged all duplicates from the set of inputs to a and b 
static bool AreDuplicate(csg::Expr a, csg::Expr b)
{
    if (a.node->op != b.node->op) {
        return false;
    }
    auto op = a.node->op;

    if (a.IsAxisOp()) {
        return true;
    }
    if (op == csg::Operator::CONST) {
        return a.node->constant == b.node->constant;
    }
    assert(a.IsInputOp());

    if (a.node->inputs.size() != b.node->inputs.size()) {
        return false;
    }
    for (size_t i = 0; i < a.node->inputs.size(); i++) {
        // here we assume that we have merged all duplicates from the set of inputs
        // and that a and b are in normal form
        if (a.node->inputs[i].node.get() != b.node->inputs[i].node.get()) {
            return false;
        }
    }
    return true;
}

csg::Expr csg::MergeDuplicates(csg::Expr root) 
{
    // First put the root in normal form.
    root = csg::NormalForm(root);

    std::vector<csg::Expr> uniqueExprs;
    return root.TopoMap<csg::Expr>([&] (csg::Expr e, std::vector<csg::Expr> inputs) {
        csg::Expr newE;
        if (e.IsConstantOp()) {
            newE.node = std::make_shared<csg::Node>(e.node->constant);
        }
        else {
            newE.node = std::make_shared<csg::Node>(e.node->op, std::move(inputs));
        }

        for (csg::Expr e2 : uniqueExprs) {
            if (AreDuplicate(newE, e2)) {
                return e2;
            }
        }
        uniqueExprs.push_back(newE);
        return newE;
    });
}


/*template <typename T>
static void VectorExtend(std::vector<T>& v1, const std::vector<T>& v2)
{
    for (const T& x : v2) {
        v1.push_back(x);
    }
}

class AffineForm
{
public:
    struct SlopeExpr
    {
        float slope;
        csg::Expr expr;
    };

    // The expression this represents is 
    // intercept + ADD_i (exprs[i].slope * exprs[i].expr).
    // All the expressions are in normal form.
    float intercept = 0;
    std::vector<SlopeExpr> exprs;
    
    AffineForm() {}
    AffineForm(float constant) : intercept(constant) {}
    // Assumes e is in normal form
    AffineForm(csg::Expr e) : intercept(0) 
    {
        exprs.push_back({ 1, e });
    }

    bool IsConstant() const 
    {
        return exprs.size() == 0;
    }

    csg::Expr ToNormFormExpr() const
    {
        csg::Expr res(intercept);
        for (const auto& e : exprs) {
            res = NormalFormStep(res + NormalFormStep(e.slope * e.expr));
        }
        return res;
    }

    static AffineForm Add(const AffineForm& a, const AffineForm& b)
    {
        AffineForm res;
        res.intercept = a.intercept + b.intercept;
        for (const auto& e : a.exprs) {
            AddSingle(res, e);
        }
        for (const auto& e : b.exprs) {
            AddSingle(res, e);
        }
        return res;
    }

    static AffineForm Multiply(const AffineForm& a, float x)
    {
        AffineForm res;
        if (x == 0) { return res; }

        res.intercept = x * a.intercept;
        for (const auto& e : a.exprs) {
            res.exprs.push_back({ x * e.slope, e.expr });
        }
        return res;
    }


private:
    static void AddSingle(AffineForm& a, const AffineForm::SlopeExpr& eB)
    {
        for (auto& eA : a.exprs) {
            if (csg::ThreeWayCompare(eA.expr, eB.expr) == 0) {
                eA.slope += eB.slope;
                return;
            }
        }
        a.exprs.push_back(eB);
    }
};


csg::Expr csg::AffineFold(csg::Expr root) 
{
    AffineForm rootAF = root.TopoMap<AffineForm>([=] (csg::Expr e, std::vector<AffineForm> inputs) {
        switch (e.node->op) {
        case csg::Operator::CONST:
            return AffineForm(e.node->constant);
        case csg::Operator::ADD: 
            assert(inputs.size() == 2);
            return AffineForm::Add(inputs[0], inputs[1]);
        case csg::Operator::SUB: 
            assert(inputs.size() == 2);
            return AffineForm::Add(inputs[0], AffineForm::Multiply(inputs[1], -1));
        case csg::Operator::MUL:
            assert(inputs.size() == 2);
            if (inputs[0].IsConstant()) { return AffineForm::Multiply(inputs[1], inputs[0].intercept); }
            if (inputs[1].IsConstant()) { return AffineForm::Multiply(inputs[0], inputs[1].intercept); }
            break;
        case csg::Operator::DIV:
            if (inputs[1].IsConstant()) { return AffineForm::Multiply(inputs[0], 1 / inputs[1].intercept); }
            break;
        case csg::Operator::NEG:
            return AffineForm::Multiply(inputs[0], -1);
        default: break;
        }
        std::vector<csg::Expr> newInputs;
        for (const AffineForm& i : inputs) {
            newInputs.push_back(i.ToNormFormExpr());
        }
        csg::Expr newE(std::make_shared<csg::Node>(e.node->op, std::move(newInputs)));
        //return AffineForm(NormalFormStep(newE));
        return AffineForm(NormalForm(ConstantFold(newE)));
    }); 
    return rootAF.ToNormFormExpr();
}*/
