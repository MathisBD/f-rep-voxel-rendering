#include "csg/expression.h"
#include <glm/glm.hpp>


bool csg::Expr::IsAxisOp() const 
{
    assert(node);
    return node->op == csg::Operator::X || 
           node->op == csg::Operator::Y ||
           node->op == csg::Operator::Z;
}

bool csg::Expr::IsConstantOp() const 
{
    assert(node);
    return node->op == csg::Operator::CONST;    
}

bool csg::Expr::IsInputOp() const 
{
    assert(node);
    return node->op == csg::Operator::SIN  ||
           node->op == csg::Operator::COS  ||
           node->op == csg::Operator::EXP  ||
           node->op == csg::Operator::NEG  ||
           node->op == csg::Operator::SQRT ||
           node->op == csg::Operator::ADD  ||
           node->op == csg::Operator::SUB  ||
           node->op == csg::Operator::MUL  ||
           node->op == csg::Operator::DIV  ||
           node->op == csg::Operator::MIN  ||
           node->op == csg::Operator::MAX;
}

csg::Expr csg::operator-(csg::Expr e)
{
    std::vector<csg::Expr> inputs = { e };
    return { .node = std::make_shared<csg::Node>(csg::Operator::NEG, std::move(inputs)) };
}


csg::Expr csg::operator+(csg::Expr e1, csg::Expr e2) 
{
    std::vector<csg::Expr> inputs = { e1, e2 };
    return { .node = std::make_shared<csg::Node>(csg::Operator::ADD, std::move(inputs)) };
}
csg::Expr csg::operator-(csg::Expr e1, csg::Expr e2) 
{
    std::vector<csg::Expr> inputs = { e1, e2 };
    return { .node = std::make_shared<csg::Node>(csg::Operator::SUB, std::move(inputs)) };
}
csg::Expr csg::operator*(csg::Expr e1, csg::Expr e2) 
{
    std::vector<csg::Expr> inputs = { e1, e2 };
    return { .node = std::make_shared<csg::Node>(csg::Operator::MUL, std::move(inputs)) };
}
csg::Expr csg::operator/(csg::Expr e1, csg::Expr e2) 
{
    std::vector<csg::Expr> inputs = { e1, e2 };
    return { .node = std::make_shared<csg::Node>(csg::Operator::DIV, std::move(inputs)) };
}


csg::Expr csg::operator+(float constant, csg::Expr e2)
{
    csg::Expr e1 = { .node = std::make_shared<csg::Node>(constant) };
    return e1 + e2;
}
csg::Expr csg::operator-(float constant, csg::Expr e2)
{
    csg::Expr e1 = { .node = std::make_shared<csg::Node>(constant) };
    return e1 - e2;
}
csg::Expr csg::operator*(float constant, csg::Expr e2)
{
    csg::Expr e1 = { .node = std::make_shared<csg::Node>(constant) };
    return e1 * e2;
}
csg::Expr csg::operator/(float constant, csg::Expr e2)
{
    csg::Expr e1 = { .node = std::make_shared<csg::Node>(constant) };
    return e1 / e2;
}


csg::Expr csg::operator+(csg::Expr e1, float constant)
{
    csg::Expr e2 = { .node = std::make_shared<csg::Node>(constant) };
    return e1 + e2;
}
csg::Expr csg::operator-(csg::Expr e1, float constant)
{
    csg::Expr e2 = { .node = std::make_shared<csg::Node>(constant) };
    return e1 - e2;
}
csg::Expr csg::operator*(csg::Expr e1, float constant)
{
    csg::Expr e2 = { .node = std::make_shared<csg::Node>(constant) };
    return e1 * e2;
}
csg::Expr csg::operator/(csg::Expr e1, float constant)
{
    csg::Expr e2 = { .node = std::make_shared<csg::Node>(constant) };
    return e1 / e2;
}


float csg::Expr::Eval(float x, float y, float z) const
{
    switch (node->op) {
    case csg::Operator::X:      return x;
    case csg::Operator::Y:      return y;
    case csg::Operator::Z:      return z;
    case csg::Operator::CONST:  return node->constant;
    case csg::Operator::SIN:    
        assert(node->inputs.size() == 1);
        return glm::sin(node->inputs[0].Eval(x, y, z));
    case csg::Operator::COS:    
        assert(node->inputs.size() == 1);
        return glm::cos(node->inputs[0].Eval(x, y, z));
    case csg::Operator::EXP:    
        assert(node->inputs.size() == 1);
        return glm::exp(node->inputs[0].Eval(x, y, z));
    case csg::Operator::NEG:    
        assert(node->inputs.size() == 1);
        return -node->inputs[0].Eval(x, y, z);
    case csg::Operator::SQRT:    
        assert(node->inputs.size() == 1);
        return glm::sqrt(node->inputs[0].Eval(x, y, z));
    case csg::Operator::ADD:    
        assert(node->inputs.size() == 2);
        return node->inputs[0].Eval(x, y, z) + node->inputs[1].Eval(x, y, z);
    case csg::Operator::SUB:    
        assert(node->inputs.size() == 2);
        return node->inputs[0].Eval(x, y, z) - node->inputs[1].Eval(x, y, z);
    case csg::Operator::MUL:    
        assert(node->inputs.size() == 2);
        return node->inputs[0].Eval(x, y, z) * node->inputs[1].Eval(x, y, z);
    case csg::Operator::DIV: 
        {   
            assert(node->inputs.size() == 2);
            float denom = node->inputs[1].Eval(x, y, z);
            assert(denom != 0.0f);
            return node->inputs[0].Eval(x, y, z) / denom;
        }
    case csg::Operator::MIN:    
        assert(node->inputs.size() == 2);
        return glm::min(node->inputs[0].Eval(x, y, z), node->inputs[1].Eval(x, y, z));
    case csg::Operator::MAX:    
        assert(node->inputs.size() == 2);
        return glm::max(node->inputs[0].Eval(x, y, z), node->inputs[1].Eval(x, y, z));
    }
    assert(false); return 0.0f;
}

static std::string OpName(csg::Operator op)
{
    switch (op) {
    case csg::Operator::X:      return "X";
    case csg::Operator::Y:      return "Y";
    case csg::Operator::Z:      return "Z";
    case csg::Operator::CONST:  return "CONST";
    case csg::Operator::SIN:    return "SIN";
    case csg::Operator::COS:    return "COS";
    case csg::Operator::EXP:    return "EXP";
    case csg::Operator::NEG:    return "NEG";
    case csg::Operator::SQRT:   return "SQRT";
    case csg::Operator::ADD:    return "ADD";
    case csg::Operator::SUB:    return "SUB";
    case csg::Operator::MUL:    return "MUL";
    case csg::Operator::DIV:    return "DIV";
    case csg::Operator::MIN:    return "MIN";
    case csg::Operator::MAX:    return "MAX";
    default: assert(false); return "";
    }
}

void csg::Expr::Print() const 
{
    assert(node);

    if (IsAxisOp()) {
        printf("0x%lx:%s", 
            (size_t)node.get(), OpName(node->op).c_str());
    }
    else if (IsConstantOp()) {
        printf("0x%lx:%s   const=%.3f", 
            (size_t)node.get(), OpName(node->op).c_str(), node->constant);
    }
    else if (IsInputOp()) {
        printf("0x%lx:%s   inputs=", 
            (size_t)node.get(), OpName(node->op).c_str());
        for (auto inp : node->inputs) {
            printf("0x%lx   ", (size_t)inp.node.get());
        }
    }
    else {
        assert(false);
    }
}

csg::Expr csg::X()
{
    std::vector<csg::Expr> inputs = {};
    return { .node = std::make_shared<csg::Node>(csg::Operator::X, std::move(inputs)) };
}

csg::Expr csg::Y()
{
    std::vector<csg::Expr> inputs = {};
    return { .node = std::make_shared<csg::Node>(csg::Operator::Y, std::move(inputs)) };
}

csg::Expr csg::Z()
{
    std::vector<csg::Expr> inputs = {};
    return { .node = std::make_shared<csg::Node>(csg::Operator::Z, std::move(inputs)) };
}

csg::Expr csg::Constant(float x)
{
    return { .node = std::make_shared<csg::Node>(x) };
}

csg::Expr csg::Sin(csg::Expr e)
{
    std::vector<csg::Expr> inputs = { e };
    return { .node = std::make_shared<csg::Node>(csg::Operator::SIN, std::move(inputs)) };
}

csg::Expr csg::Cos(csg::Expr e)
{
    std::vector<csg::Expr> inputs = { e };
    return { .node = std::make_shared<csg::Node>(csg::Operator::COS, std::move(inputs)) };
}

csg::Expr csg::Exp(csg::Expr e)
{
    std::vector<csg::Expr> inputs = { e };
    return { .node = std::make_shared<csg::Node>(csg::Operator::EXP, std::move(inputs)) };
}

csg::Expr csg::Sqrt(csg::Expr e)
{
    std::vector<csg::Expr> inputs = { e };
    return { .node = std::make_shared<csg::Node>(csg::Operator::SQRT, std::move(inputs)) };
}

csg::Expr csg::Min(csg::Expr e1, csg::Expr e2) 
{
    std::vector<csg::Expr> inputs = { e1, e2 };
    return { .node = std::make_shared<csg::Node>(csg::Operator::MIN, std::move(inputs)) };
}

csg::Expr csg::Max(csg::Expr e1, csg::Expr e2) 
{
    std::vector<csg::Expr> inputs = { e1, e2 };
    return { .node = std::make_shared<csg::Node>(csg::Operator::MAX, std::move(inputs)) };
}