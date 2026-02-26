#pragma once

namespace SPM
{
    class LevelSet
{
    private:
    int nlevels;
    int nodeNum;
public:
    int *level_ptr;
    int *permutation;
    int *permToOrig;
    int *nodeToLevel;   // 1-based
    

    LevelSet();
    LevelSet(int nlevels, int nodeNum,int *level_ptr, int *nodeToLevel, int *permutation, int *permToOrig);
    ~LevelSet();
    

    void alloc();   // allocate memory space for level_ptr, permutation, permToOrig, nodeToLevel
    void initalize(int nodeNum, int nlevels);    // nodeNum and levels，execute alloc function


    void dealloc();

    int getLevels() const ;
    void setLevels(int nlevels);

    int getNodeNum() const;
    void setNodeNum(int nodeNum);

    bool equal(const LevelSet &levelset);

    void printLevelPtr() const;
    void printNodeToLevel() const;
    void printPermutation() const;
};


} // namespace SPM

