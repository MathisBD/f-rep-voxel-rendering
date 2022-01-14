#include "csg/simplifier.h"


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


csg::Expr csg::UnfoldArithNodes(csg::Expr root) 
{
    return root.TopoMap<csg::Expr>([=] (csg::Expr e, std::vector<csg::Expr> inputs) {
        switch (e.node->op) {
        case csg::Operator::ADD: {
            assert(inputs.size() >= 2);
            csg::Expr e2 = inputs[0] + inputs[1];
            for (size_t i = 2; i < inputs.size(); i++) {
                e2 = e2 + inputs[i];
            }
            return e2;
        }
        case csg::Operator::MUL: {
            assert(inputs.size() >= 2);
            csg::Expr e2 = inputs[0] * inputs[1];
            for (size_t i = 2; i < inputs.size(); i++) {
                e2 = e2 * inputs[i];
            }
            return e2;
        }
        default: return csg::Expr(std::make_shared<csg::Node>(e.node->op, std::move(inputs)));
        }
    }); 
}

template <typename T>
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
}