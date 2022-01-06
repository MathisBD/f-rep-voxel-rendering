#include "engine/csg_expression.h"
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
    return node->op == csg::Operator::SIN ||
           node->op == csg::Operator::COS ||
           node->op == csg::Operator::ADD ||
           node->op == csg::Operator::SUB ||
           node->op == csg::Operator::MUL ||
           node->op == csg::Operator::DIV;
}

csg::Expr csg::Expr::operator+(const csg::Expr other) const 
{
    std::vector<csg::Expr> inputs = { *this, other };
    return { .node = std::make_shared<csg::Node>(csg::Operator::ADD, std::move(inputs)) };
}

csg::Expr csg::Expr::operator-(const csg::Expr other) const
{
    std::vector<csg::Expr> inputs = { *this, other };
    return { .node = std::make_shared<csg::Node>(csg::Operator::SUB, std::move(inputs)) };
}

csg::Expr csg::Expr::operator*(const csg::Expr other) const
{
    std::vector<csg::Expr> inputs = { *this, other };
    return { .node = std::make_shared<csg::Node>(csg::Operator::MUL, std::move(inputs)) };
}

csg::Expr csg::Expr::operator/(const csg::Expr other) const
{
    std::vector<csg::Expr> inputs = { *this, other };
    return { .node = std::make_shared<csg::Node>(csg::Operator::DIV, std::move(inputs)) };
}

float csg::Expr::Eval(float x, float y, float z) const
{
    switch (node->op) {
    case csg::Operator::X:      return x;
    case csg::Operator::Y:      return y;
    case csg::Operator::Z:      return z;
    case csg::Operator::CONST:  return node->constant;
    case csg::Operator::SIN:    return glm::sin(node->inputs[0].Eval(x, y, z));
    case csg::Operator::COS:    return glm::cos(node->inputs[0].Eval(x, y, z));
    case csg::Operator::ADD:    return node->inputs[0].Eval(x, y, z) + node->inputs[1].Eval(x, y, z);
    case csg::Operator::SUB:    return node->inputs[0].Eval(x, y, z) - node->inputs[1].Eval(x, y, z);
    case csg::Operator::MUL:    return node->inputs[0].Eval(x, y, z) * node->inputs[1].Eval(x, y, z);
    case csg::Operator::DIV:    return node->inputs[0].Eval(x, y, z) / node->inputs[1].Eval(x, y, z);
    default: assert(false); return 0.0f;
    }
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
    case csg::Operator::ADD:    return "ADD";
    case csg::Operator::SUB:    return "SUB";
    case csg::Operator::MUL:    return "MUL";
    case csg::Operator::DIV:    return "DIV";
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

