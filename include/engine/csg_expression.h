#pragma once
#include <vector>
#include <stdint.h>
#include <memory>
#include <assert.h>



namespace csg
{
    class Node;

    // The user should be manipulating expressions, not nodes,
    // while building the tree.
    // An expression is just a wrapper around a node.
    // It is designed to be passed by value, and handles the deletion
    // of its node.
    class Expression 
    {
    public:
        std::shared_ptr<Node> node;

        Expression operator+(const Expression other) const;
        Expression operator-(const Expression other) const;
        Expression operator*(const Expression other) const;
        Expression operator/(const Expression other) const;
        float Eval(float x, float y, float z) const;
    };

    csg::Expression X();
    csg::Expression Y();
    csg::Expression Z();
    csg::Expression Constant(float x);
    csg::Expression Sin(csg::Expression e);
    csg::Expression Cos(csg::Expression e);


    enum class Operator
    {
        // Nullary operators (no inputs)
        X, Y, Z, CONSTANT,
        // Unary operators
        SIN, COS, //EXP, NEG
        // Binary operators
        ADD, SUB, MUL, DIV, //MIN, MAX
    };

    // Contains all the information for a CSG operation.
    class Node
    {
    public:
        Operator op;
        // A node that contains a constant doesn't have any inputs,
        // so we can use a union here.
        union {
            std::vector<Expression> inputs;
            float constant;
        };

        Node(float constant) 
        {
            this->op = Operator::CONSTANT;
            this->constant = constant;
        }

        Node(Operator op, std::vector<Expression>&& inputs) 
        {
            assert(op != Operator::CONSTANT);
            this->op = op;
            this->inputs = inputs;
        }

        ~Node() {}
    };
}
