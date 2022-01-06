#pragma once
#include <vector>
#include <stdint.h>
#include "engine/csg_expression.h"
#include <unordered_map>
#include <functional>



namespace csg
{
    class Tape
    {
    public:
        enum Op {
            LOAD_CONST = 0,
            SIN, COS,
            ADD, SUB, MUL, DIV  
        };

        // A single tape instruction.
        struct Instr 
        {
            uint16_t op;
            uint16_t outSlot;

            union {
                // LOAD_CONST
                float constant;
                // slot ops
                struct { uint16_t inSlotA; uint16_t inSlotB; };
            };
        };

        std::vector<Instr> instructions;
    

        Tape();
        Tape(Expr e);

        static std::string OpName(uint16_t op);
        void Print() const;
        float Eval(float x, float y, float z) const;
    private:
        // The unique x, y and z nodes in the expression DAG.
        Expr m_x, m_y, m_z;
        // The root of the expression DAG.
        Expr m_result;

        // A topological sort of the expression DAG;
        std::vector<Expr> m_exprs;
        // The index in the topo sort of each DAG node.
        std::unordered_map<csg::Node*, uint32_t> m_nodeIdx;
        // The liveliness of each DAG node.
        // The liveliness is the index of the last DAG node (in the topo sort)
        // that uses this node as input.
        // The liveliness can be -1, which indicates the node's output is never used.
        std::vector<int> m_liveliness;

        std::vector<Expr> m_slots;

        // Applies f to every expression in the DAG of e,
        // and guarantees that a node is visited after all its children
        // have been visited.
        void TopoMap(Expr e, const std::function<void(Expr)>& f);

        Expr MergeAxesNodes(Expr e);
        void TopoSort();

        void ComputeLiveliness();

        uint32_t GetFreeSlot();
        uint32_t GetCurrentSlot(Expr e);
        void ReleaseSlot(Expr e);

        uint16_t TapeOpFromExprOp(Operator op);

        void BuildInstrs();
    };
}