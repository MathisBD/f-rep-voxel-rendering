#include "csg/expression.h"
#include <glm/glm.hpp>
#include <unordered_set>
#include <unordered_map>


csg::Node::Node(float constant) 
{
    this->op = Operator::CONST;
    this->constant = constant;
}

csg::Node::Node(Operator op, std::vector<Expr>&& inputs) 
{
    assert(op != Operator::CONST);
    this->op = op;
    this->inputs = inputs;
}


bool csg::Expr::IsAxisOp() const 
{
    assert(node);
    return node->op == csg::Operator::X || 
           node->op == csg::Operator::Y ||
           node->op == csg::Operator::Z ||
           node->op == csg::Operator::T;
}

bool csg::Expr::IsConstantOp() const 
{
    assert(node);
    return node->op == csg::Operator::CONST;    
}

bool csg::Expr::IsConstantOp(float constant) const 
{
    assert(node);
    return node->op == csg::Operator::CONST && node->constant == constant;    
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
    return { std::make_shared<csg::Node>(csg::Operator::NEG, std::move(inputs)) };
}


csg::Expr csg::operator+(csg::Expr e1, csg::Expr e2) 
{
    std::vector<csg::Expr> inputs = { e1, e2 };
    return { std::make_shared<csg::Node>(csg::Operator::ADD, std::move(inputs)) };
}
csg::Expr csg::operator-(csg::Expr e1, csg::Expr e2) 
{
    std::vector<csg::Expr> inputs = { e1, e2 };
    return { std::make_shared<csg::Node>(csg::Operator::SUB, std::move(inputs)) };
}
csg::Expr csg::operator*(csg::Expr e1, csg::Expr e2) 
{
    std::vector<csg::Expr> inputs = { e1, e2 };
    return { std::make_shared<csg::Node>(csg::Operator::MUL, std::move(inputs)) };
}
csg::Expr csg::operator/(csg::Expr e1, csg::Expr e2) 
{
    std::vector<csg::Expr> inputs = { e1, e2 };
    return { std::make_shared<csg::Node>(csg::Operator::DIV, std::move(inputs)) };
}


float csg::ApplyOperator(csg::Operator op, std::vector<float> args) 
{
    switch (op) {
    case csg::Operator::SIN:    
        assert(args.size() == 1);
        return glm::sin(args[0]);
    case csg::Operator::COS:    
        assert(args.size() == 1);
        return glm::cos(args[0]);
    case csg::Operator::EXP:    
        assert(args.size() == 1);
        return glm::exp(args[0]);
    case csg::Operator::NEG:    
        assert(args.size() == 1);
        return -args[0];
    case csg::Operator::SQRT:    
        assert(args.size() == 1);
        return glm::sqrt(args[0]);
    case csg::Operator::ADD: 
        {   
            assert(args.size() >= 2);
            float res = 0;
            for (float x : args) {
                res += x;
            }
            return res;
        }
    case csg::Operator::SUB:    
        assert(args.size() == 2);
        return args[0] - args[1];
    case csg::Operator::MUL: 
        {
            assert(args.size() >= 2);
            float res = 1;
            for (float x : args) {
                res *= x;
            }
            return res;
        }
    case csg::Operator::DIV: 
        {   
            assert(args.size() == 2);
            assert(args[1] != 0.0f);
            return args[0] / args[1];
        }
    case csg::Operator::MIN:    
        assert(args.size() == 2);
        return glm::min(args[0], args[1]);
    case csg::Operator::MAX:    
        assert(args.size() == 2);
        return glm::max(args[0], args[1]);
    default: assert(false); return 0.0f;
    }
}

float csg::Expr::Eval(float x, float y, float z, float t) const
{
    switch (node->op) {
    case csg::Operator::X:      return x;
    case csg::Operator::Y:      return y;
    case csg::Operator::Z:      return z;
    case csg::Operator::T:      return t;
    case csg::Operator::CONST:  return node->constant;
    default:
        std::vector<float> args;
        for (csg::Expr i : node->inputs) {
            args.push_back(i.Eval(x, y, z, t));
        }
        return csg::ApplyOperator(node->op, args);
    }
}

static std::string OpName(csg::Operator op)
{
    switch (op) {
    case csg::Operator::X:      return "X";
    case csg::Operator::Y:      return "Y";
    case csg::Operator::Z:      return "Z";
    case csg::Operator::T:      return "T";
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
    return { std::make_shared<csg::Node>(csg::Operator::X, std::move(inputs)) };
}

csg::Expr csg::Y()
{
    std::vector<csg::Expr> inputs = {};
    return { std::make_shared<csg::Node>(csg::Operator::Y, std::move(inputs)) };
}

csg::Expr csg::Z()
{
    std::vector<csg::Expr> inputs = {};
    return { std::make_shared<csg::Node>(csg::Operator::Z, std::move(inputs)) };
}

csg::Expr csg::T()
{
    std::vector<csg::Expr> inputs = {};
    return { std::make_shared<csg::Node>(csg::Operator::T, std::move(inputs)) };
}

csg::Expr csg::Constant(float x)
{
    return { x };
}

csg::Expr csg::Sin(csg::Expr e)
{
    std::vector<csg::Expr> inputs = { e };
    return { std::make_shared<csg::Node>(csg::Operator::SIN, std::move(inputs)) };
}

csg::Expr csg::Cos(csg::Expr e)
{
    std::vector<csg::Expr> inputs = { e };
    return { std::make_shared<csg::Node>(csg::Operator::COS, std::move(inputs)) };
}

csg::Expr csg::Exp(csg::Expr e)
{
    std::vector<csg::Expr> inputs = { e };
    return { std::make_shared<csg::Node>(csg::Operator::EXP, std::move(inputs)) };
}

csg::Expr csg::Sqrt(csg::Expr e)
{
    std::vector<csg::Expr> inputs = { e };
    return { std::make_shared<csg::Node>(csg::Operator::SQRT, std::move(inputs)) };
}

csg::Expr csg::Min(csg::Expr e1, csg::Expr e2) 
{
    std::vector<csg::Expr> inputs = { e1, e2 };
    return { std::make_shared<csg::Node>(csg::Operator::MIN, std::move(inputs)) };
}

csg::Expr csg::Max(csg::Expr e1, csg::Expr e2) 
{
    std::vector<csg::Expr> inputs = { e1, e2 };
    return { std::make_shared<csg::Node>(csg::Operator::MAX, std::move(inputs)) };
}


csg::Expr csg::Expr::operator()(csg::Expr newX, csg::Expr newY, csg::Expr newZ, csg::Expr newT) const 
{
    return TopoMap<csg::Expr>([=] (csg::Expr e, std::vector<csg::Expr> inputs) {
        switch (e.node->op) {
        case csg::Operator::X: return newX;
        case csg::Operator::Y: return newY;
        case csg::Operator::Z: return newZ;
        case csg::Operator::T: return newT;
        case csg::Operator::CONST: return e;
        default: 
            assert(e.IsInputOp());
            return csg::Expr(std::make_shared<csg::Node>(e.node->op, std::move(inputs)));
        }
    });
}


void csg::Expr::TopoIter(const std::function<void(csg::Expr)>& f) const
{
    std::unordered_set<csg::Node*> visited; 
    std::function<void(csg::Expr)> dfs = [&] (csg::Expr e) {
        if (visited.find(e.node.get()) == visited.end()) {
            visited.insert(e.node.get());

            if (e.IsInputOp()) {
                for (csg::Expr child : e.node->inputs) {
                    dfs(child);
                }
            }
            f(e);
        }
    };
    dfs(*this);
}

std::string csg::Expr::ToDotGraph() const 
{
    DotGraph graph(true);

    TopoMap<int>([&] (csg::Expr e, std::vector<int> inputs) {
        std::string label;
        if (e.IsConstantOp()) {
            char buf[64];
            sprintf(buf, "%.2f", e.node->constant);
            label = std::string(buf);
        }
        else {
            label = OpName(e.node->op);
        }

        int id = graph.AddNode(label);
        for (int cid : inputs) {
            graph.AddEdge(id, cid);
        }
        return id;
    });

    return graph.Build();
}