#include "engine/csg_expression.h"
#include <glm/glm.hpp>


csg::Expression csg::Expression::operator+(const csg::Expression other) const 
{
    std::vector<csg::Expression> inputs = { *this, other };
    return { .node = std::make_shared<csg::Node>(csg::Operator::ADD, std::move(inputs)) };
}

csg::Expression csg::Expression::operator-(const csg::Expression other) const
{
    std::vector<csg::Expression> inputs = { *this, other };
    return { .node = std::make_shared<csg::Node>(csg::Operator::SUB, std::move(inputs)) };
}

csg::Expression csg::Expression::operator*(const csg::Expression other) const
{
    std::vector<csg::Expression> inputs = { *this, other };
    return { .node = std::make_shared<csg::Node>(csg::Operator::MUL, std::move(inputs)) };
}

csg::Expression csg::Expression::operator/(const csg::Expression other) const
{
    std::vector<csg::Expression> inputs = { *this, other };
    return { .node = std::make_shared<csg::Node>(csg::Operator::DIV, std::move(inputs)) };
}

float csg::Expression::Eval(float x, float y, float z) const
{
    switch (node->op) {
    case csg::Operator::X: return x;
    case csg::Operator::Y: return y;
    case csg::Operator::Z: return z;
    case csg::Operator::CONSTANT: return node->constant;
    case csg::Operator::SIN: return glm::sin(node->inputs[0].Eval(x, y, z));
    case csg::Operator::COS: return glm::cos(node->inputs[0].Eval(x, y, z));
    case csg::Operator::ADD: return node->inputs[0].Eval(x, y, z) + node->inputs[1].Eval(x, y, z);
    case csg::Operator::SUB: return node->inputs[0].Eval(x, y, z) - node->inputs[1].Eval(x, y, z);
    case csg::Operator::MUL: return node->inputs[0].Eval(x, y, z) * node->inputs[1].Eval(x, y, z);
    case csg::Operator::DIV: return node->inputs[0].Eval(x, y, z) / node->inputs[1].Eval(x, y, z);
    }
    assert(false);
    return 0.0f;
}

csg::Expression csg::X()
{
    std::vector<csg::Expression> inputs = {};
    return { .node = std::make_shared<csg::Node>(csg::Operator::X, std::move(inputs)) };
}

csg::Expression csg::Y()
{
    std::vector<csg::Expression> inputs = {};
    return { .node = std::make_shared<csg::Node>(csg::Operator::Y, std::move(inputs)) };
}

csg::Expression csg::Z()
{
    std::vector<csg::Expression> inputs = {};
    return { .node = std::make_shared<csg::Node>(csg::Operator::Z, std::move(inputs)) };
}

csg::Expression csg::Constant(float x)
{
    return { .node = std::make_shared<csg::Node>(x) };
}

csg::Expression csg::Sin(csg::Expression e)
{
    std::vector<csg::Expression> inputs = { e };
    return { .node = std::make_shared<csg::Node>(csg::Operator::SIN, std::move(inputs)) };
}

csg::Expression csg::Cos(csg::Expression e)
{
    std::vector<csg::Expression> inputs = { e };
    return { .node = std::make_shared<csg::Node>(csg::Operator::COS, std::move(inputs)) };
}

