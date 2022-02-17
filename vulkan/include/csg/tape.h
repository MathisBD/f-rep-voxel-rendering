#pragma once
#include <vector>
#include <stdint.h>
#include "csg/expression.h"
#include <unordered_map>
#include <functional>



namespace csg
{
    class Tape
    {
    public:
        enum Op {
            LOAD_CONST = 0,
            SIN = 1, 
            COS = 2,
            ADD = 3, 
            SUB = 4, 
            MUL = 5, 
            DIV = 6,
            MIN = 7,
            MAX = 8,
            EXP = 9,
            NEG = 10,
            SQRT = 11,
            COPY = 12,
            _OP_COUNT_
        };

        // A single tape instruction.
        struct Instr 
        {
            uint8_t op;
            uint8_t outSlot;
            uint8_t inSlotA;
            uint8_t inSlotB;
        };

        std::vector<Instr> instructions;
        std::vector<float> constantPool;

        Tape();
        Tape(Expr e);

        static std::string OpName(uint16_t op);
        uint32_t GetSlotCount() const;
        void Print(bool detailed = false) const;
        float Eval(float x, float y, float z, float t) const;
    private:
        // The root of the simplified expression DAG.
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

        // The index of each constant in the constant pool.
        std::unordered_map<float, uint32_t> m_constantIdx;

        std::vector<Expr> m_slots;
        
        Expr Simplify(Expr e);
        void BuildConstantPool();
        void TopoSort();

        void ComputeLiveliness();

        uint32_t GetFreeSlot();
        uint32_t GetCurrentSlot(Expr e);
        void ReleaseSlot(Expr e);

        uint16_t TapeOpFromExprOp(Operator op);

        void BuildInstrs();
    };
}