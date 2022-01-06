#include "engine/csg_tape.h"
#include <unordered_set>
#include <unordered_map>
#include <functional>
#include <algorithm>
#include <stdio.h>
#include <glm/glm.hpp>


csg::Tape::Tape() {}

csg::Tape::Tape(csg::Expr e) 
{
    m_result = MergeAxesNodes(e);
    TopoSort();
    ComputeLiveliness();
    BuildInstrs();
}

csg::Expr csg::Tape::MergeAxesNodes(csg::Expr root) 
{
    // Maps old nodes to new nodes.
    std::unordered_map<csg::Node*, csg::Expr> newExpr;

    std::function<void(csg::Expr)> copyExpr = [&] (csg::Expr e) 
    {
        csg::Expr newE;
        switch (e.node->op) {
        case csg::Operator::X:
            if (!m_x.node) { m_x = e; }
            newE = m_x;
            break;
        case csg::Operator::Y:
            if (!m_y.node) { m_y = e; }
            newE = m_y;
            break; 
        case csg::Operator::Z:
            if (!m_z.node) { m_z = e; }
            newE = m_z;
            break;
        case csg::Operator::CONST:
            newE.node = std::make_shared<csg::Node>(e.node->constant);
            break;
        default:
            assert(e.IsInputOp());
            std::vector<csg::Expr> newInputs;
            for (auto child : e.node->inputs) {
                newInputs.push_back(newExpr[child.node.get()]);
            }
            newE.node = std::make_shared<csg::Node>(e.node->op, std::move(newInputs));
            break;
        }
        newExpr[e.node.get()] = newE;
    };

    TopoMap(root, copyExpr);
    return newExpr[root.node.get()];
}

void csg::Tape::TopoSort() 
{
    TopoMap(m_result, [=] (csg::Expr e) {
        m_exprs.push_back(e);
    });

    // Calculate the index of each node in the topo sort.
    for (uint32_t i = 0; i < m_exprs.size(); i++) {
        m_nodeIdx[m_exprs[i].node.get()] = i;
    }    
}

void csg::Tape::TopoMap(csg::Expr root, const std::function<void(csg::Expr)>& f) 
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

    dfs(root);
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
    }    
    assert(false);
    return 0;
}


void csg::Tape::BuildInstrs() 
{
    assert(m_x.node->op == csg::Operator::X);
    assert(m_y.node->op == csg::Operator::Y);
    assert(m_z.node->op == csg::Operator::Z);

    // Initially the x, y and z values occupy the first three slots
    m_slots.push_back(m_x);
    m_slots.push_back(m_y);
    m_slots.push_back(m_z);

    for (uint32_t i = 0; i < m_exprs.size(); i++) {
        csg::Expr e = m_exprs[i];
        csg::Tape::Instr inst;

        if (e.IsAxisOp()) {
            assert(e.node.get() == m_x.node.get() || 
                   e.node.get() == m_y.node.get() || 
                   e.node.get() == m_z.node.get());
            // We don't build an instruction for these expressions.
            continue;
        }
        else if (e.IsConstantOp()) {
            inst.constant = e.node->constant;
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
    default: assert(false); return "";
    }
}

void csg::Tape::Print() const 
{
    printf("[+] Tape : instr count=%lu   slot cout = %lu\n", 
        instructions.size(), m_slots.size());
    for (uint32_t i = 0; i < instructions.size(); i++) {
        Instr inst = instructions[i];
        switch (inst.op) {
        case LOAD_CONST: 
            printf("\t%10s  out=%u   const=%.3f\n", 
                OpName(inst.op).c_str(), inst.outSlot, inst.constant);
            break;
        case SIN:
        case COS:
            printf("\t%10s  out=%u   in=%u\n", 
                OpName(inst.op).c_str(), inst.outSlot, inst.inSlotA);
            break;
        case ADD:
        case SUB:
        case MUL:
        case DIV:
            printf("\t%10s  out=%u   inA=%u   inB=%u\n", 
                OpName(inst.op).c_str(), inst.outSlot, inst.inSlotA, inst.inSlotB);
            break;
        default:
            assert(false);
        }
    }
}

float csg::Tape::Eval(float x, float y, float z) const 
{
    std::vector<float> slots(m_slots.size());
    slots[0] = x;
    slots[1] = y;
    slots[2] = z;

    for (auto i : instructions) {
        switch (i.op) {
        case Op::LOAD_CONST: slots[i.outSlot] = i.constant; break;
        case Op::SIN:        slots[i.outSlot] = glm::sin(slots[i.inSlotA]); break;
        case Op::COS:        slots[i.outSlot] = glm::cos(slots[i.inSlotA]); break;
        case Op::ADD:        slots[i.outSlot] = slots[i.inSlotA] + slots[i.inSlotB]; break;
        case Op::SUB:        slots[i.outSlot] = slots[i.inSlotA] - slots[i.inSlotB]; break;
        case Op::MUL:        slots[i.outSlot] = slots[i.inSlotA] * slots[i.inSlotB]; break;
        case Op::DIV:        slots[i.outSlot] = slots[i.inSlotA] / slots[i.inSlotB]; break;
        default: assert(false);
        }
    }
    return slots[instructions.back().outSlot];
}