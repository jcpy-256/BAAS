
#include <iostream>
#include "LevelSet.hpp"
#include "MarcoUtils.hpp"
#include "spmUtils.hpp"

namespace SPM
{
    LevelSet::LevelSet() : level_ptr(nullptr), nodeToLevel(nullptr), permutation(nullptr), permToOrig(nullptr) {}

    LevelSet::LevelSet(int nlevels, int nodeNum, int *level_ptr, int *nodeToLevel, int *permutation, int *permToOrig)
    {
        this->nlevels = nlevels;
        this->nodeNum = nodeNum;
        this->level_ptr = level_ptr;
        this->nodeToLevel = nodeToLevel;
        this->permutation = permutation;
        this->permToOrig = permToOrig;
    }

    LevelSet::~LevelSet()
    {
        dealloc();
    }

    void LevelSet::initalize(int nodeNum, int nlevls)
    {
        setNodeNum(nodeNum);
        setLevels(nlevels);
        alloc();
    }

    void LevelSet::alloc()
    {
        this->level_ptr = MALLOC(int, nlevels + 1);
        this->nodeToLevel = MALLOC(int, nodeNum);
        this->permutation = MALLOC(int, nodeNum);
        this->permToOrig = MALLOC(int, nodeNum);
        CHECK_POINTER(this->level_ptr);
        CHECK_POINTER(this->nodeToLevel);
        CHECK_POINTER(this->permutation);
        CHECK_POINTER(this->permToOrig);
    }

    void LevelSet::dealloc()
    {
        FREE(this->level_ptr);
        FREE(this->nodeToLevel);
        FREE(this->permToOrig);
        FREE(this->permutation);
        // FREE(nodeToLevel);
    }

    int LevelSet::getLevels() const { return nlevels; }
    void LevelSet::setLevels(int nlevels) { this->nlevels = nlevels; }

    int LevelSet::getNodeNum() const
    {
        return this->nodeNum;
    };
    void LevelSet::setNodeNum(int nodeNum)
    {
        this->nodeNum = nodeNum;
    };

    bool LevelSet::equal(const LevelSet &L)
    {
        if (nodeNum != L.getNodeNum())
        {
            printf("nodeNum is not euqal\n");
            return false;
        }
        // CHECK_POINTER(this->level_ptr)
        if (nlevels != L.getLevels())
        {
            printf("nlevels is not equal\n");
            return false;
        }
        if (ivectorEqual(level_ptr, L.level_ptr, nlevels) == false)
        {
            printf("level ptr is not equal\n");
            return false;
        }

        if (ivectorEqual(nodeToLevel, L.nodeToLevel, nodeNum) == false)
        {
            printf("node to level mapping is not equal\n");
            return false;
        }
        if (ivectorEqual(permutation, L.permutation, nodeNum) == false)
        {
            printf("permutation array is not equal\n");
            return false;
        }
        if (ivectorEqual(permToOrig, L.permToOrig, nodeNum) == false)
        {
            printf("permutation to origin is not equal\n");
            return false;
        }

        return true;
    }
    void LevelSet::printLevelPtr() const
    {
        for (int i = 0; i < nlevels + 1; i++)
        {
            printf("level ptr %d:%d  ", i, level_ptr[i]);
        }
        printf("\n");
    }
    void LevelSet::printNodeToLevel() const
    {
        for (int i = 0; i < nodeNum; i++)
        {
            printf("node %d : level %d", i, nodeToLevel[i]);
        }
        printf("\n");
    }
    void LevelSet::printPermutation() const
    {
        for (int i = 0; i < nodeNum; i++)
        {
            printf(" %d th : node %d", i, permutation[i]);
        }
        printf("\n");
    }

} // namespace SPM
