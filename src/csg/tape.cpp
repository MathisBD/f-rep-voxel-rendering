#include "csg/tape.h"
#include <unordered_set>
#include <unordered_map>
#include <functional>
#include <algorithm>
#include <stdio.h>
#include <glm/glm.hpp>
#include <fstream>
#include "csg/simplifier.h"


csg::Tape::Tape() {}

csg::Tape::Tape(csg::Expr e) 
{
    m_result = Simplify(e);
    TopoSort();
    BuildConstantPool();
    ComputeLiveliness();
    BuildInstrs();
}

csg::Expr csg::Tape::Simplify(csg::Expr root) 
{
    csg::Expr e = root;
    e = csg::ConstantFold(e);
    e = csg::MergeDuplicates(e);
    return e;
}

void csg::Tape::TopoSort() 
{
    m_result.TopoIter([=] (csg::Expr e) {
        m_exprs.push_back(e);
    });

    // Calculate the index of each node in the topo sort.
    for (uint32_t i = 0; i < m_exprs.size(); i++) {
        m_nodeIdx[m_exprs[i].node.get()] = i;
    }    
}


void csg::Tape::BuildConstantPool() 
{
    for (auto e : m_exprs) {
        if (e.IsConstantOp()) {
            float c = e.node->constant;
            if (m_constantIdx.find(c) == m_constantIdx.end()) {
                constantPool.push_back(c);
                m_constantIdx[c] = constantPool.size() - 1;
            }
        }
    }
}


static int max(int a, int b)
{
    return a > b ? a : b;
}

// The liveliness of a node is the index of the last
// node that uses its result.
// The liveliness can be < 0 if no one ever uses the node's result.
void csg::Tape::ComputeLiveliness() 
{
    m_liveliness = std::vector<int>(m_exprs.size(), -1);
    for (uint32_t i = 0; i < m_exprs.size(); i++) {
        if (m_exprs[i].IsInputOp()) {
            for (auto input : m_exprs[i].node->inputs) {
                uint32_t inputIdx = m_nodeIdx[input.node.get()];
                // The max() is not strictly necessary, 
                // but it clarifies what we are doing.
                m_liveliness[inputIdx] = max(i, m_liveliness[inputIdx]);
            }
        }
    }
}

uint32_t csg::Tape::GetFreeSlot() 
{
    for (uint32_t i = 0; i < m_slots.size(); i++) {
        if (!m_slots[i].node) {
            return i;
        }
    }
    // No free slot : add a new one
    assert(m_slots.size() < 256);
    m_slots.push_back(csg::Expr());
    return m_slots.size() - 1;
}

uint32_t csg::Tape::GetCurrentSlot(csg::Expr e) 
{
    for (uint32_t i = 0; i < m_slots.size(); i++) {
        if (m_slots[i].node.get() == e.node.get()) {
            return i;
        }
    }
    assert(false);
    return 0;
}

void csg::Tape::ReleaseSlot(csg::Expr e)
{
    for (uint32_t i = 0; i < m_slots.size(); i++) {
        if (m_slots[i].node.get() == e.node.get()) {
            m_slots[i] = csg::Expr();
            // check e is not in any other slot
            for (auto e2 : m_slots) {
                assert(e2.node.get() != e.node.get());
            }
            return;
        }
    }
    // It is okay to call ReleaseSlot() even if e
    // doesn't occupy a slot anymore.
    // This is to handle gracefully instructions that have
    // both inputs in the same slot.
}

uint16_t csg::Tape::TapeOpFromExprOp(csg::Operator op) 
{
    switch (op) {
    case csg::Operator::X:
    case csg::Operator::Y:
    case csg::Operator::Z:
    case csg::Operator::T:
        // These expressions don't build any instruction.
        assert(false);
        return 0;
    case csg::Operator::CONST:  return csg::Tape::Op::LOAD_CONST;
    case csg::Operator::SIN:    return csg::Tape::Op::SIN;
    case csg::Operator::COS:    return csg::Tape::Op::COS;
    case csg::Operator::ADD:    return csg::Tape::Op::ADD;
    case csg::Operator::SUB:    return csg::Tape::Op::SUB;
    case csg::Operator::MUL:    return csg::Tape::Op::MUL;
    case csg::Operator::DIV:    return csg::Tape::Op::DIV;
    case csg::Operator::MIN:    return csg::Tape::Op::MIN;
    case csg::Operator::MAX:    return csg::Tape::Op::MAX;
    case csg::Operator::EXP:    return csg::Tape::Op::EXP;
    case csg::Operator::NEG:    return csg::Tape::Op::NEG;
    case csg::Operator::SQRT:   return csg::Tape::Op::SQRT;
    }    
    assert(false);
    return 0;
}


void csg::Tape::BuildInstrs() 
{
    csg::Expr x, y, z, t;
    for (auto& e : m_exprs) {
        switch (e.node->op) {
        case csg::Operator::X: x = e; break;
        case csg::Operator::Y: y = e; break;
        case csg::Operator::Z: z = e; break;
        case csg::Operator::T: t = e; break;
        default: break;
        }
    }

    // Initially the x, y, z and t values occupy the first three slots
    m_slots.push_back(x);
    m_slots.push_back(y);
    m_slots.push_back(z);
    m_slots.push_back(t);

    for (uint32_t i = 0; i < m_exprs.size(); i++) {
        csg::Expr e = m_exprs[i];
        csg::Tape::Instr inst = {};

        if (e.IsAxisOp()) {
            assert(e.node.get() == x.node.get() || 
                   e.node.get() == y.node.get() || 
                   e.node.get() == z.node.get() ||
                   e.node.get() == t.node.get());
            // We don't build an instruction for these expressions.
            continue;
        }
        else if (e.IsConstantOp()) {
            inst.inSlotA = m_constantIdx[e.node->constant];
        }
        else if (e.IsInputOp() && e.node->inputs.size() == 1) {
            csg::Expr inpA = e.node->inputs[0];
            inst.inSlotA = GetCurrentSlot(inpA);
            
            assert(m_liveliness[m_nodeIdx[inpA.node.get()]] >= (int)i);
            if (m_liveliness[m_nodeIdx[inpA.node.get()]] == (int)i) {
                ReleaseSlot(inpA);
            }
        }
        else if (e.IsInputOp() && e.node->inputs.size() == 2) {
            csg::Expr inpA = e.node->inputs[0];
            csg::Expr inpB = e.node->inputs[1];

            inst.inSlotA = GetCurrentSlot(e.node->inputs[0]);
            inst.inSlotB = GetCurrentSlot(e.node->inputs[1]);
            
            assert(m_liveliness[m_nodeIdx[inpA.node.get()]] >= (int)i);
            if (m_liveliness[m_nodeIdx[inpA.node.get()]] == (int)i) {
                ReleaseSlot(inpA);
            }
            assert(m_liveliness[m_nodeIdx[inpB.node.get()]] >= (int)i);
            if (m_liveliness[m_nodeIdx[inpB.node.get()]] == (int)i) {
                ReleaseSlot(inpB);
            }
        }
        else {
            assert(false);
        }
        
        // We release the slots BEFORE acquiring one for the output.
        // This way a single instruction can read and write to the same slot.
        inst.outSlot = GetFreeSlot();
        m_slots[inst.outSlot] = e;

        inst.op = TapeOpFromExprOp(e.node->op);
        instructions.push_back(inst);
    }
    // Check the output slot is slot 0
    assert(instructions.back().outSlot == 0);
    // Check we didn't overflow the 8bit quantities
    assert(m_slots.size() <= 256);
    assert(Op::_OP_COUNT_ <= 256);
    assert(constantPool.size() <= 256);
}

std::string csg::Tape::OpName(uint16_t op)
{
    switch (op) {
    case LOAD_CONST: return "CONST";
    case SIN: return "SIN";
    case COS: return "COS";
    case ADD: return "ADD";
    case SUB: return "SUB";
    case MUL: return "MUL";
    case DIV: return "DIV";
    case MIN: return "MIN";
    case MAX: return "MAX";
    case EXP: return "EXP";
    case NEG: return "NEG";
    case SQRT: return "SQRT";
    case COPY: return "COPY";
    default: assert(false); return "";
    }
}

void csg::Tape::Print() const 
{
    printf("[+] Tape : instr count=%lu   slot cout = %lu\n", 
        instructions.size(), m_slots.size());
    for (uint32_t i = 0; i < instructions.size(); i++) {
        Instr inst = instructions[i];
        printf("\t%2u   %10s  out=%u   inA=%u   inB=%u\n", 
            i, OpName(inst.op).c_str(), inst.outSlot, inst.inSlotA, inst.inSlotB);
    }
    printf("[+] Constant Pool:\n");
    for (uint32_t i = 0; i < constantPool.size(); i++) {
        printf("\t%2u   %4.2f\n", i, constantPool[i]);
    }
}

float csg::Tape::Eval(float x, float y, float z, float t) const 
{
    std::vector<float> slots(m_slots.size());
    slots[0] = x;
    slots[1] = y;
    slots[2] = z;
    slots[3] = t;

    for (auto i : instructions) {
        switch (i.op) {
        case Op::LOAD_CONST: slots[i.outSlot] = constantPool[i.inSlotA]; break;
        case Op::SIN:        slots[i.outSlot] = glm::sin(slots[i.inSlotA]); break;
        case Op::COS:        slots[i.outSlot] = glm::cos(slots[i.inSlotA]); break;
        case Op::ADD:        slots[i.outSlot] = slots[i.inSlotA] + slots[i.inSlotB]; break;
        case Op::SUB:        slots[i.outSlot] = slots[i.inSlotA] - slots[i.inSlotB]; break;
        case Op::MUL:        slots[i.outSlot] = slots[i.inSlotA] * slots[i.inSlotB]; break;
        case Op::DIV:        slots[i.outSlot] = slots[i.inSlotA] / slots[i.inSlotB]; break;
        case Op::MIN:        slots[i.outSlot] = glm::min(slots[i.inSlotA], slots[i.inSlotB]); break;
        case Op::MAX:        slots[i.outSlot] = glm::max(slots[i.inSlotA], slots[i.inSlotB]); break;
        case Op::EXP:        slots[i.outSlot] = glm::exp(slots[i.inSlotA]); break;
        case Op::NEG:        slots[i.outSlot] = -slots[i.inSlotA]; break;
        case Op::SQRT:       slots[i.outSlot] = glm::sqrt(slots[i.inSlotA]); break;
        case Op::COPY:       slots[i.outSlot] = slots[i.inSlotA]; break;
        default: assert(false);
        }
    }
    assert(instructions.back().outSlot == 0);
    return slots[0];
}