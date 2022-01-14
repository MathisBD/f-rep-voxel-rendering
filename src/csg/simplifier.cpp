#include "csg/simplifier.h"
#include <algorithm>


csg::Expr csg::MergeAxes(csg::Expr root) 
{
    csg::Expr x, y, z, t;
    return root.TopoMap<csg::Expr>([&] (csg::Expr e, std::vector<csg::Expr> inputs) {
        switch (e.node->op) {
        case csg::Operator::X:
            if (!x.node) { x = e; }
            return x;
        case csg::Operator::Y:
            if (!y.node) { y = e; }
            return y;
        case csg::Operator::Z:
            if (!z.node) { z = e; }
            return z;
        case csg::Operator::T:
            if (!t.node) { t = e; }
            return t;
        case csg::Operator::CONST:
            return e;
        default: 
            assert(e.IsInputOp());
            return csg::Expr(std::make_shared<csg::Node>(e.node->op, std::move(inputs)));
        }
    });
}

csg::Expr csg::ConstantFold(csg::Expr root) 
{    
    return root.TopoMap<csg::Expr>([=] (csg::Expr e, std::vector<csg::Expr> inputs) {
        if (!e.IsInputOp()) {
            return e;
        }

        std::vector<float> constants;
        for (auto i : inputs) {
            if (i.IsConstantOp()) {
                constants.push_back(i.node->constant);
            }
        }
        // All inputs are constant
        if (constants.size() == inputs.size()) {
            float res = csg::ApplyOperator(e.node->op, constants);
            return csg::Expr(res);
        }

        // For add : merge all the constants
        if (constants.size() > 0 && 
            (e.node->op == csg::Operator::ADD || e.node->op == csg::Operator::MUL)) 
        {
            assert(inputs.size() >= 2);
            float id = e.node->op == csg::Operator::ADD ? 0 : 1;
            
            float c = id;
            std::vector<csg::Expr> newInputs;
            for (const auto& inp : inputs) {
                if (inp.IsConstantOp()) {
                    c = ApplyOperator(e.node->op, { c, inp.node->constant });
                }
                else {
                    newInputs.push_back(inp);
                }
            }
            if (c != id) {
                newInputs.push_back(csg::Expr(c));
            }
            assert(newInputs.size() >= 1);
            if (newInputs.size() == 1) {
                return newInputs[0];
            }
            inputs = newInputs;
        }

        // Some inputs are constant :
        // we can replace some operations with simpler ones.
        switch (e.node->op) {
        case csg::Operator::MUL:
            assert(inputs.size() >= 2);
            for (const auto& inp : inputs) {
                if (inp.IsConstantOp(0)) {
                    return csg::Exp(0);
                }
            }
            break;
        case csg::Operator::SUB:
            assert(inputs.size() == 2);
            if (inputs[0].IsConstantOp(0)) { return -inputs[1]; }
            if (inputs[1].IsConstantOp(0)) { return inputs[0]; }
            if (inputs[1].IsConstantOp())  { return inputs[0] + (-inputs[1].node->constant); }
            break;
        case csg::Operator::DIV:
            assert(inputs.size() == 2);
            if (inputs[0].IsConstantOp(0))        { return csg::Expr(0); }
            if (inputs[1].IsConstantOp(1))        { return inputs[0]; }
            if (e.node->inputs[1].IsConstantOp()) { return inputs[0] * (1 / inputs[1].node->constant); }
            break;
        case csg::Operator::NEG:
            if (inputs[0].node->op == csg::Operator::NEG) { return inputs[0].node->inputs[0]; }
            break;
        default: break;
        }

        return csg::Expr(std::make_shared<csg::Node>(e.node->op, std::move(inputs)));
    });   
}


template <typename T>
static void VectorExtend(std::vector<T>& v1, const std::vector<T>& v2)
{
    for (const T& x : v2) {
        v1.push_back(x);
    }
}

/*class AffineForm
{
public:
    struct SlopeExpr
    {
        float slope;
        csg::Expr expr;
    };

    // The expression this represents is 
    // intercept + ADD_i (exprs[i].slope * exprs[i].expr)
    float intercept;
    std::vector<SlopeExpr> exprs;
    
    AffineForm() 
    {
        intercept = 0;
    }
    AffineForm(float intercept) 
    {
        this->intercept = intercept;
    }

    csg::Expr ToExpr() const
    {
        csg::Expr res(intercept);
        for (const auto& e : exprs) {
            res = res + (e.slope * e.expr);
        }
        return res;
    }

    static ANF Multiply(const ANF& a, const ANF& b)
    {
        ANF result;
        for (size_t i = 0; i < a.exprs.size(); i++) {
            for (size_t j = 0; j < b.exprs.size(); j++) {
                std::vector<csg::Expr> es;
                VectorExtend(es, a.exprs[i]);
                VectorExtend(es, b.exprs[j]);
                result.exprs.push_back(es);
            }
        }
        return result;
    }

    static AffineForm Opposite(const AffineForm& a)
    {
        AffineForm result;
        result.intercept = -a.intercept;
        for (size_t i = 0; i < a.exprs.size(); i++) {
            std::vector<csg::Expr> es;
            

            for (size_t j = 0; j < a.exprs[i].size(); j++) {
                if (j == idx) {
                    es.push_back(-a.exprs[i][j]);
                }
                else {
                    es.push_back(a.exprs[i][j]);
                }
            }
            result.exprs.push_back(es);
        }
        return result;
    }
};


csg::Expr csg::ArithNormalForm(csg::Expr root) 
{
    AffineForm rootAF = root.TopoMap<AffineForm>([=] (csg::Expr e, std::vector<AffineForm> inputs) {
        AffineForm af;
        switch (e.node->op) {
        case csg::Operator::ADD: 
            for (const ANF& i : inputs) {
                VectorExtend(anf.exprs, i.exprs);
            }
            break;
        case csg::Operator::MUL:
            anf = inputs[0];
            for (size_t i = 1; i < inputs.size(); i++) {
                anf = ANF::Multiply(anf, inputs[i]);
            }
            break;
        case csg::Operator::NEG:
            assert(inputs.size() == 1);
            anf = ANF::Negate(inputs[0]);
            break;
        case csg::Operator::CONST:
            anf = ANF(e);
            break;
        default:
            std::vector<csg::Expr> eInputs;
            for (const ANF& i : inputs) {
                eInputs.push_back(i.ToExpr());
            }
            anf = ANF(csg::Expr(std::make_shared<csg::Node>(e.node->op, std::move(eInputs))));
            break;
        }
    }); 
    return rootAF.ToExpr();
}*/

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


struct CompareFunctor
{
    int operator()(const csg::Expr a, const csg::Expr b)
    {
        return csg::ThreeWayCompare(a, b);
    }
};

csg::Expr csg::NormalForm(csg::Expr root) 
{
    return root.TopoMap<csg::Expr>([=] (csg::Expr e, std::vector<csg::Expr> inputs) {
        if (!e.IsInputOp()) {
            return e;
        } 

        switch (e.node->op) {
        // These are commutative operations :
        // we have to sort their operands
        case csg::Operator::ADD:
        case csg::Operator::MUL:
        case csg::Operator::MIN:
        case csg::Operator::MAX:
            std::sort(inputs.begin(), inputs.end(), 
                [] (const csg::Expr a, const csg::Expr b) 
            {
                return ThreeWayCompare(a, b) <= 0;
            });
            break;
        default: break;
        }
        return csg::Expr(std::make_shared<csg::Node>(e.node->op, std::move(inputs)));
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