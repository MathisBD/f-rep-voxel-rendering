#pragma once
#include <vector>
#include <stdint.h>
#include <memory>
#include <assert.h>
#include <functional>



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

        Expr() : node(nullptr) {};
        Expr(std::shared_ptr<Node> node_) : node(node_) {};
        Expr(float constant) : node(std::make_shared<csg::Node>(constant)) {};

        bool IsAxisOp() const;
        bool IsConstantOp() const;
        bool IsInputOp() const;

        // Returns true if this is a constant op 
        // and the constant is equal to the argument.
        bool IsConstantOp(float constant) const;
    
        // build a new expression DAG where x is replaced with newX,
        // y with newY and z with newZ.
        Expr operator()(Expr newX, Expr newY, Expr newZ, Expr newT) const;

        float Eval(float x, float y, float z, float t) const;
        void Print() const;

        // Applies f to every expression in the DAG of e,
        // and guarantees that a node is visited after all its children
        // have been visited.
        void TopoIter(const std::function<void(Expr)>& f) const;
        // Similar to TopoIter, but f takes as input : the expression e,
        // and the result of f applied to the children of e,
        // and returns a new expression.
        Expr TopoMap(const std::function<Expr(Expr, std::vector<Expr>)>& f) const;
    };

    Expr operator-(Expr e);
    Expr operator+(Expr e1, Expr e2);
    Expr operator-(Expr e1, Expr e2);
    Expr operator*(Expr e1, Expr e2);
    Expr operator/(Expr e1, Expr e2);

    Expr X();
    Expr Y();
    Expr Z();
    Expr T();
    Expr Constant(float x);
    Expr Sin(Expr e);
    Expr Cos(Expr e);
    Expr Min(Expr e1, Expr e2);
    Expr Max(Expr e1, Expr e2);
    Expr Exp(Expr e);
    Expr Sqrt(Expr e);
    
    enum class Operator
    {
        // Nullary operators (no inputs)
        X, Y, Z, T, CONST,
        // Unary operators
        SIN, COS, EXP, NEG, SQRT,
        // Binary operators
        ADD, SUB, MUL, DIV, MIN, MAX
    };

    float ApplyOperator(Operator op, std::vector<float> args);

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
