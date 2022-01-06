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
    class Expr 
    {
    public:
        std::shared_ptr<Node> node = nullptr;

        bool IsAxisOp() const;
        bool IsConstantOp() const;
        bool IsInputOp() const;
        
        Expr operator+(const Expr other) const;
        Expr operator-(const Expr other) const;
        Expr operator*(const Expr other) const;
        Expr operator/(const Expr other) const;

        float Eval(float x, float y, float z) const;
        void Print() const;
    };

    // These avoid the need for writing e + csg::Constant(1.0f).
    // We can just write e + 10.0f and the float will be automatically
    // wrapped in an expression.
    Expr operator+(float constant, const Expr e);
    Expr operator-(float constant, const Expr e);
    Expr operator*(float constant, const Expr e);
    Expr operator/(float constant, const Expr e);

    Expr operator+(const Expr e, float constant);
    Expr operator-(const Expr e, float constant);
    Expr operator*(const Expr e, float constant);
    Expr operator/(const Expr e, float constant);


    csg::Expr X();
    csg::Expr Y();
    csg::Expr Z();
    csg::Expr Constant(float x);
    csg::Expr Sin(csg::Expr e);
    csg::Expr Cos(csg::Expr e);


    enum class Operator
    {
        // Nullary operators (no inputs)
        X, Y, Z, CONST,
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
        std::vector<Expr> inputs;
        float constant;

        Node(float constant) 
        {
            this->op = Operator::CONST;
            this->constant = constant;
        }

        Node(Operator op, std::vector<Expr>&& inputs) 
        {
            assert(op != Operator::CONST);
            this->op = op;
            this->inputs = inputs;
        }

        ~Node() {}
    };
}
