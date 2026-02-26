/**
 * @file graph opertation base on these basic fotmats in this folder
 */

#include <vector>
#include <cstdlib>
#include <cstring>
#include <queue>
#include "DAG.hpp"
#include "MarcoUtils.hpp"
#include "spmUtils.hpp"
#include "MathUtils.hpp"

namespace SPM
{
    DAG::DAG() : n(0), edges(0) {
                 };
    DAG::~DAG()
    {
        dealloc();
    }

    void DAG::alloc()
    {
        DAG_vw.resize(edges);
        DAG_set.resize(edges);
        // DAG_ptr.resize(n + 1);
        DAG_ptr.resize(n + 1, 0);
        prealloc = true;
    }

    DAG::DAG(int n, int edges, const DAG_MAT mat /*DAG_CSC*/)
    {
        // this->m = m;
        this->n = n;
        this->edges = edges;
        this->format = mat;
        alloc();
    }

    DAG::DAG(int n, int edges, const int *set, const int *ptr, const DAG_MAT mat /*DAG_CSC*/)
    {
        assert(ptr[n] == edges);
        assert(set != nullptr && ptr != nullptr);
        this->n = n;
        this->edges = edges;
        this->format = mat;
        alloc();
        std::copy(ptr, ptr + n + 1, DAG_ptr.begin());
        std::copy(set, set + edges, DAG_set.begin()); // copyt iterator data
    }

    DAG::DAG(int n, int edges, const int *set, const int *ptr, const double *weight, const DAG_MAT mat /*DAG_CSC*/) : DAG(n, edges, mat)
    {
        //  DAG_vw.resize(edges);
        // DAG(n, edges); // alloc space
        std::copy(ptr, ptr + n + 1, DAG_ptr.begin());
        std::copy(set, set + edges, DAG_set.begin()); // copyt iterator data
        std::copy(weight, weight + edges, DAG_vw.begin());
        // this->format = mat;
    }

    DAG::DAG(DAG &dag) : DAG(dag.n, dag.edges)
    {
        // n = dag.n;
        // edges = dag.edges;
        // DAG(n, edges);
        // deep copy
        DAG_set = dag.DAG_set;
        DAG_ptr = dag.DAG_ptr;
        DAG_vw = dag.DAG_vw;
        format = dag.format;
    }

    void DAG::dealloc()
    {
        n = 0;
        edges = 0;
        // destory vectory
        DAG_set.clear();
        DAG_ptr.clear();
        DAG_vw.clear();
        // free memory
        DAG_set.shrink_to_fit();
        DAG_ptr.shrink_to_fit();
        DAG_vw.shrink_to_fit();
    }
    bool DAG::isNullDAG()
    {
        if (DAG_ptr.size() == 0 || DAG_set.size() == 0 || n == 0)
        {
            return true;
        }
        return false;
    }

    void DAG::updateEdges()
    {
        assert(DAG_ptr.back() == DAG_set.size());
        edges = DAG_ptr.back();
    }

    int DAG::getEdges() const
    {
        // if (edges != DAG_ptr.back())
        //     this->updateEdges();
        return DAG_ptr.back();
    }

    int DAG::getNodeNum() const
    {
        return n;
    }

    void DAG::setNodeNum(const int nodeNum)
    {
        this->n = nodeNum;
    }

    // 对DAG 进行 two-hop  transitive reduction 并返回一个pruned DAG
    void DAG::partialSparsification_CSC(const int n, const int edges_not_prune, const int *DAG_ptr_not_prune, const int *DAG_set_not_prune,
                                        std::vector<int> &DAG_ptr, std::vector<int> &DAG_set, bool cliqueSimplification /*=true*/)
    {

        if (cliqueSimplification)
        {
            // find cliques -> after simplification they are just chains
            // For now it only support lower matrices
            auto Lp = DAG_ptr_not_prune;
            auto Li = DAG_set_not_prune;
            int limit = n;
            // It is a CSC/CSR like format
            std::vector<int> clique_ptr;
            clique_ptr.reserve(n);
            clique_ptr.push_back(0);
            std::vector<int> node_to_clique(n);
#pragma omp parallel
            {
                int bins = omp_get_num_threads();
                int tid = omp_get_thread_num();
                int start_col = (n / bins) * tid;
                int end_col = (n / bins) * (tid + 1);
                std::vector<int> clique_ptr_per_thread;
                if (tid == bins - 1)
                {
                    end_col = n;
                }
                clique_ptr_per_thread.reserve(end_col - start_col);
                clique_ptr_per_thread.push_back(start_col);
                for (int col = start_col; col < end_col;)
                {
                    int width = 1;
                    int first = col;
                    int last = col + 1;
                    while (last < n && width < limit)
                    {
                        int diff = last - first;
                        // compare number of off diagonals
                        int off_last = Lp[last + 1] - Lp[last];
                        int off_first = Lp[first + 1] - Lp[first] - diff;
                        // If you don't understand this line
                        // get back and read about clique .. don't waste your time
                        if (off_first != off_last)
                            break;
                        assert(off_first > 0 && off_last > 0);
                        // now examine column pattern
                        bool can_form_clique = true;
                        for (int row = 0; row < off_first; row++)
                        {
                            if (Li[Lp[first] + row + diff] != Li[Lp[last] + row])
                            {
                                can_form_clique = false;
                                break;
                            }
                        }
                        if (can_form_clique)
                        {
                            // Increase the width of the clique
                            width++;
                            // Lets check the next row
                            last++;
                        }
                        else
                        {
                            // Lets go to the next row, this clique is done
                            break;
                        }
                    }
                    // The starting point of the next clique
                    col = first + width;
                    assert(col <= n);
                    clique_ptr_per_thread.push_back(col);
                    assert(Lp[first + 1] - Lp[first] - width >= 0);
                }

                for (int i = 0; i < clique_ptr_per_thread.size() - 1; i++)
                {
                    int start_node = clique_ptr_per_thread[i];
                    int end_node = clique_ptr_per_thread[i + 1];
                    for (int j = start_node; j < end_node; j++)
                    {
                        node_to_clique[j] = i + start_col;
                    }
                }
            }

            int back = 0;
            int node_cnt = 0;
            for (int i : node_to_clique)
            {
                if (i != back)
                {
                    clique_ptr.push_back(node_cnt);
                }
                node_cnt++;
                back = i;
            }
            clique_ptr.push_back(node_cnt);
            assert(clique_ptr.back() == n);
            // Creating the prune DAG based on the clique pointers
            std::vector<int> DAG_ptr_no_clique(n + 1, 0);
            std::vector<int> DAG_set_no_clique;
            DAG_set_no_clique.reserve(edges_not_prune);
            int edge_cnt = 0;
            for (int i = 0; i < clique_ptr.size() - 1; i++)
            {
                for (int j = clique_ptr[i]; j < clique_ptr[i + 1] - 1; j++)
                {
                    DAG_ptr_no_clique[j] = edge_cnt;
                    DAG_set_no_clique.push_back(j);     // clique内部相连。添加本节点的DAG对角位置
                    DAG_set_no_clique.push_back(j + 1); // 添加和下一个节点相连的位置
                    edge_cnt += 2;
                }
                int last_node = clique_ptr[i + 1] - 1;
                DAG_ptr_no_clique[last_node] = edge_cnt;
                for (int j = DAG_ptr_not_prune[last_node]; j < DAG_ptr_not_prune[last_node + 1]; j++)
                {
                    DAG_set_no_clique.push_back(DAG_set_not_prune[j]); // 添加最后一个节点的所有DAG列数据
                    edge_cnt++;
                }
            }
            DAG_ptr_no_clique[n] = edge_cnt;

            std::vector<int> deleted_edge(edge_cnt, 0);
#pragma omp parallel for
            for (int i = 0; i < n; i++)
            {
                int start_i_child_ptr = DAG_ptr_no_clique[i] + 1;
                int end_i_child_ptr = DAG_ptr_no_clique[i + 1];
                for (int k_ptr = DAG_ptr_no_clique[i] + 1; k_ptr < DAG_ptr_no_clique[i + 1]; k_ptr++)
                {
                    int k = DAG_set_no_clique[k_ptr]; // k: Child of i
                    int l2 = DAG_ptr_no_clique[k] + 1;
                    int u2 = DAG_ptr_no_clique[k + 1]; //[l2 - u2) Child range of k
                    int l1 = start_i_child_ptr;
                    int u1 = end_i_child_ptr;
                    while (l1 < u1 && l2 < u2)
                    { // two pointer scan in sequential array to get same row index
                        if (DAG_set_no_clique[l1] == DAG_set_no_clique[l2])
                        {
                            deleted_edge[l1] = 1;
                            l1++;
                            l2++;
                        }
                        else if (DAG_set_no_clique[l1] < DAG_set_no_clique[l2])
                        {
                            l1++;
                        }
                        else
                        {
                            l2++;
                        }
                    }
                }
            }

            DAG_ptr.resize(n + 1, 0);
            DAG_set.resize(0);
            DAG_set.reserve(edge_cnt);
            int edge_counter = 0;
            for (int i = 0; i < n; i++)
            {
                for (int k_ptr = DAG_ptr_no_clique[i]; k_ptr < DAG_ptr_no_clique[i + 1]; k_ptr++)
                {
                    if (!deleted_edge[k_ptr])
                    {
                        int k = DAG_set_no_clique[k_ptr]; // Child of i
                        DAG_set.push_back(k);
                        edge_counter++;
                    }
                }
                DAG_ptr[i + 1] = edge_counter;
            }
        }
        else
        {
            // std::cout << "SLOW Sparsification" << std::endl;
            std::vector<int> deleted_edge(edges_not_prune, 0);
#pragma omp parallel for
            for (int i = 0; i < n; i++)
            {
                std::vector<int> child(DAG_ptr_not_prune[i + 1] - (DAG_ptr_not_prune[i] + 1));
                int cnt = 0;
                int start_k_ptr = DAG_ptr_not_prune[i] + 1;
                for (int k_ptr = DAG_ptr_not_prune[i] + 1; k_ptr < DAG_ptr_not_prune[i + 1]; k_ptr++)
                {
                    int k = DAG_set_not_prune[k_ptr]; // Child of i
                    child[cnt++] = k;
                }

                for (int k_ptr = DAG_ptr_not_prune[i] + 1; k_ptr < DAG_ptr_not_prune[i + 1]; k_ptr++)
                {
                    int k = DAG_set_not_prune[k_ptr]; // Child of i
                    for (int j_ptr = DAG_ptr_not_prune[k] + 1; j_ptr < DAG_ptr_not_prune[k + 1]; j_ptr++)
                    {
                        int j = DAG_set_not_prune[j_ptr]; // Child of k
                        auto iter = std::lower_bound(child.begin(), child.end(), j);
                        if (iter != child.end() && *iter <= j)
                        {
                            auto id = std::distance(child.begin(), iter);
                            deleted_edge[start_k_ptr + id] = 1;
                        }
                    }
                }
            }

            DAG_ptr.resize(n + 1, 0);
            DAG_set.resize(0);
            DAG_set.reserve(edges_not_prune);
            int edge_counter = 0;
            for (int i = 0; i < n; i++)
            {
                for (int k_ptr = DAG_ptr_not_prune[i]; k_ptr < DAG_ptr_not_prune[i + 1]; k_ptr++)
                {
                    if (!deleted_edge[k_ptr])
                    {
                        int k = DAG_set_not_prune[k_ptr]; // Child of i
                        DAG_set.push_back(k);
                        edge_counter++;
                    }
                }
                DAG_ptr[i + 1] = edge_counter;
            }
        }
    }

    // the DAG format is not a standward lower triangular in some possible
    void DAG::findLevelsPostOrder(LevelSet *&level)
    {
        int *nodeToLevel_tmp = MALLOC(int, n);
        int *permToOrig_tmp = MALLOC(int, n);
        std::fill_n(nodeToLevel_tmp, n, 1);

        int *level_ptr_tmp = MALLOC(int, n + 1);
        memset(level_ptr_tmp, 0, (n + 1) * sizeof(int));
        int nlevels = 0;

        if (this->format == DAG_MAT::DAG_CSC) // csc format
        {
            DAG_levelSet_CSC(n, DAG_ptr.data(), DAG_set.data(), nlevels, level_ptr_tmp, nodeToLevel_tmp, permToOrig_tmp);
        }
        else if (this->format == DAG_MAT::DAG_CSR) // csr format
        {
            // std::cerr << "Invalid DAG format!" << endl;
            DAG_levelSet_CSR(n, DAG_ptr.data(), DAG_set.data(), nlevels, level_ptr_tmp, nodeToLevel_tmp, permToOrig_tmp);
        }

        // get permutation by sorting with reverser permutation
        int *permutation_tmp = MALLOC(int, n);                     // 1
//         int *permToOrig_alias = MALLOC(int, n);                    // 2
//         memcpy(permToOrig_alias, permToOrig_tmp, n * sizeof(int)); // 2
// #pragma omp parallel for schedule(static)
//         for (int i = 0; i < n; i++)
//         {
//             permutation_tmp[i] = i; // 1
//         }
//         quicksort_pair(permToOrig_alias, permutation_tmp, 0, n - 1); // 1 2
        
        getInversePerm(permutation_tmp, permToOrig_tmp, n);
        // FREE(permToOrig_alias);

        int *level_ptr_alias = MALLOC(int, nlevels + 1);
        memcpy(level_ptr_alias, level_ptr_tmp, (nlevels + 1) * sizeof(int));
        FREE(level_ptr_tmp);

        level->level_ptr = level_ptr_alias;
        level->nodeToLevel = nodeToLevel_tmp;
        level->permToOrig = permToOrig_tmp;   // 1
        level->permutation = permutation_tmp; // 2
        level->setLevels(nlevels);
        level->setNodeNum(this->getNodeNum());
    }

    // the DAG format is a standward lower triangular matirx
    void DAG::findLevels(LevelSet *&level)
    {

        {
            // initialize nodeToLevel
            int *nodeToLevel_tmp = MALLOC(int, n);
            std::fill_n(nodeToLevel_tmp, n, 1);

            if (this->format == DAG_MAT::DAG_CSC) // csc format
            {
                for (int j = 0; j < n; j++) // col
                {
                    for (int i = DAG_ptr[j] + 1; i < DAG_ptr[j + 1]; i++)
                    {
                        int rowidx = DAG_set[i];                                                             // row
                        assert(rowidx > j);                                                                  // assert lower triangular
                        nodeToLevel_tmp[rowidx] = std::max(nodeToLevel_tmp[j] + 1, nodeToLevel_tmp[rowidx]); // depedencices's level + 1
                    }
                }
            }
            else if (this->format == DAG_MAT::DAG_CSR) // csr format
            {
                for (int i = 0; i < n; i++) // row
                {
                    for (int j = DAG_ptr[i]; j < DAG_ptr[i + 1] - 1; j++)
                    {
                        int colidx = DAG_set[j];                                                        // col
                        assert(i > colidx);                                                             // assert lower triangular
                        nodeToLevel_tmp[i] = std::max(nodeToLevel_tmp[colidx] + 1, nodeToLevel_tmp[i]); // depedencices's level + 1
                    }
                }
            }

            // scan for level ptr
            int *level_ptr_tmp = MALLOC(int, n + 1);
            std::fill_n(level_ptr_tmp, n + 1, 0);
            int max_level = 1;
            for (int i = 0; i < n; i++)
            {
                max_level = nodeToLevel_tmp[i] > max_level ? nodeToLevel_tmp[i] : max_level;
                assert(nodeToLevel_tmp[i] <= n);
                level_ptr_tmp[nodeToLevel_tmp[i]]++;
            }
            prefixSumSingle(level_ptr_tmp, max_level + 1); // exclusive scan
            assert(level_ptr_tmp[max_level] == getNodeNum());

            // get reverse permutation
            int *permToOrig_tmp = MALLOC(int, n); // 2
            for (int i = 0; i < n; i++)
            {
                int level = nodeToLevel_tmp[i];               // level 1-based
                permToOrig_tmp[level_ptr_tmp[level - 1]] = i; // 2
                level_ptr_tmp[level - 1]++;
            }
            // shift left
            for (int i = max_level; i > 0; i--)
            {
                level_ptr_tmp[i] = level_ptr_tmp[i - 1];
            }
            level_ptr_tmp[0] = 0;

            // get permutation by sorting with reverser permutation
            int *permutation_tmp = MALLOC(int, n);                     // 1
            int *permToOrig_alias = MALLOC(int, n);                    // 2
            memcpy(permToOrig_alias, permToOrig_tmp, n * sizeof(int)); // 2
#pragma omp parallel for schedule(static)
            for (int i = 0; i < n; i++)
            {
                permutation_tmp[i] = i; // 1
            }
            quicksort_pair(permToOrig_alias, permutation_tmp, 0, n - 1); // 1 2
            FREE(permToOrig_alias);                                      // 2

            int *level_ptr_alias = MALLOC(int, max_level + 1);
            memcpy(level_ptr_alias, level_ptr_tmp, (max_level + 1) * sizeof(int));
            FREE(level_ptr_tmp);

            level->level_ptr = level_ptr_alias;
            level->nodeToLevel = nodeToLevel_tmp;
            level->permToOrig = permToOrig_tmp;   // 1
            level->permutation = permutation_tmp; // 2
            level->setLevels(max_level);
            level->setNodeNum(this->getNodeNum());
        }
    }

    DAG *DAG::inverseDAG(const DAG &dag, bool isSort /*true */)
    {
        int n = dag.getNodeNum();
        int edges = dag.getEdges();
        bool hasWeight = false;
        if (!dag.DAG_vw.empty())
        {
            hasWeight = true;
        }
        // dag.getEdges();

        // using reference to avoid data copy and operate data in original space
        DAG *dag_inv = new DAG(n, edges, dag.format);
        std::vector<int> &dag_ptr_inv = dag_inv->DAG_ptr;
        std::vector<int> &dag_set_inv = dag_inv->DAG_set;
        std::vector<double> &dag_vw_inv = dag_inv->DAG_vw;

        const std::vector<int> &dag_ptr = dag.DAG_ptr;
        const std::vector<int> &dag_set = dag.DAG_set;
        const std::vector<double> &dag_vw = dag.DAG_vw;

        // transpose csc to csr, and the csr format used as a dag in csc is the inversed DAG of the former
        for (int i = 0; i < edges; i++)
        {
            int rowidx = dag.DAG_set[i];
            dag_ptr_inv[rowidx]++;
        }

        vectorExclusiveScan(dag_ptr_inv.data(), 0, n);
        assert(dag_ptr_inv[n] == edges);

        // //data move
        if (hasWeight)
        {
            for (int i = 0; i < n; i++) // col
            {
                for (int j = dag_ptr[i]; j < dag_ptr[i + 1]; j++) // row
                {
                    int row = dag_set[j];
                    dag_set_inv[dag_ptr_inv[row]] = i;
                    dag_vw_inv[dag_ptr_inv[row]] = dag_vw[j];
                    dag_ptr_inv[row]++;
                }
            }
        }
        else
        {
            for (int i = 0; i < n; i++) // col
            {
                for (int j = dag_ptr[i]; j < dag_ptr[i + 1]; j++) // row
                {
                    int row = dag_set[j];
                    dag_set_inv[dag_ptr_inv[row]] = i;
                    dag_vw_inv[dag_ptr_inv[row]] = 0;
                    dag_ptr_inv[row]++;
                }
            }
        }

        // shift ptr
        for (int i = n; i > 0; i--)
        {
            dag_ptr_inv[i] = dag_ptr_inv[i - 1];
        }
        dag_ptr_inv[0] = 0;

#pragma omp parallel for
        for (int i = 0; i < n; i++)
        {
            quicksort_pair(dag_set_inv.data(), dag_vw_inv.data(), dag_ptr_inv[i], dag_ptr_inv[i + 1] - 1);
        }

        // for (int i = 0; i < dag_set_inv.size(); i++)
        // {
        //     printf("%d ", dag_set_inv[i]);
        // }
        // printf("\n");
        // for (int i = 0; i < dag_ptr_inv.size(); i++)
        // {
        //     printf("%d ", dag_ptr_inv[i]);
        // }
        // printf("\n");
        return dag_inv;
    }

    /**
     * @param level_ptr: allocated (n + 1 int array )
     * @param node_to_level: allocated
     * @param node_grouped_by_level: allocated
     *
     */
    void DAG_levelSet_CSC(int n, const int *colptr, const int *rowidx, int &nlevel, int *level_ptr, int *node_to_level, int *node_grouped_by_level)
    {
        int edge_counter = colptr[n];

        int *inDegree = MALLOC(int, n);
        memset(inDegree, 0, n * sizeof(int));
        // memset(level_ptr, 0, (n+1) * sizeof(n));

        for (int i = 0; i < edge_counter; i++)
        {
            inDegree[rowidx[i]]++;
        }

        int lvl = 0;
        int front = 0, rear = 0;
        // add all zero indgree node to queue
        for (int i = 0; i < n; i++)
        {
            if (inDegree[i] == 1)
                node_grouped_by_level[rear++] = i;
        }
        while (front < rear && rear <= n)
        {
            int rear_back = rear;
            lvl++;
            // traverse current level ,and add new node to queue
            while (front < rear_back && front < n) // current level: add currnt level node to level
            {
                int current = node_grouped_by_level[front++];
                node_to_level[current] = lvl;
                for (int child_ptr = colptr[current]; child_ptr < colptr[current + 1]; child_ptr++)
                {
                    int node = rowidx[child_ptr];
                    if (node != current) // ship diagonal element
                    {
                        inDegree[node]--;
                        if (inDegree[node] == 1) // the next level node must be the child of current level node
                        {
                            node_grouped_by_level[rear++] = node;
                        }
                    }
                }
            }
            level_ptr[lvl] = front;
        }
        assert(rear == n);
        nlevel = lvl;

        assert(level_ptr[nlevel] = n);
        assert(isPerm(node_grouped_by_level, n));

// sort node grouped by level with node index
#pragma omp parallel for schedule(static)
        for (int l = 0; l < nlevel; l++)
        {
            std::sort(&node_grouped_by_level[level_ptr[l]], &node_grouped_by_level[level_ptr[l + 1]]);
        }
    }

    // 0-based indexing
    void DAG_levelSet_CSR(int n, const int *rowptr, const int *colidx, int &nlevel, int *level_ptr, int *node_to_level, int *node_grouped_by_level)
    {
        // const int
        std::vector<std::list<int>> nodeChildren(n);
        int *inDegree = MALLOC(int, n);
        memset(inDegree, 0, n * sizeof(int));
        for (int i = 0; i < n; i++) // the first children is itself
        {
            inDegree[i] = rowptr[i + 1] - rowptr[i];
            for (int j = rowptr[i]; j < rowptr[i + 1]; j++)
            {
                int col = colidx[j];
                nodeChildren[col].push_back(i);
            }
        }

        int lvl = 0;
        int front = 0, rear = 0;
        // add all zero indgree node to queue
        for (int i = 0; i < n; i++)
        {
            if (inDegree[i] == 1)
                node_grouped_by_level[rear++] = i;
        }

        // front pointer traverses all nodes
        while (front < rear && rear <= n)
        {
            int rear_back = rear;
            lvl++;
            // traverse current level ,and add new node to queue
            while (front < rear_back && front < n) // current level: add currnt level node to level
            {
                int current = node_grouped_by_level[front++];
                node_to_level[current] = lvl;
                // iterate through nodeChild[current]
                for (auto &children : nodeChildren[current])
                {
                    if (children != current)
                    {
                        inDegree[children]--;
                        if (inDegree[children] == 1) // the next level node must be the child of current level node
                        {
                            node_grouped_by_level[rear++] = children;
                        }
                    }
                }
            }
            level_ptr[lvl] = front;
        }
        assert(rear == n);
        nlevel = lvl;

        assert(level_ptr[nlevel] = n);
        assert(isPerm(node_grouped_by_level, n));

// sort node grouped by level with node index
#pragma omp parallel for schedule(static)
        for (int l = 0; l < nlevel; l++)
        {
            std::sort(&node_grouped_by_level[level_ptr[l]], &node_grouped_by_level[level_ptr[l + 1]]);
        }

        FREE(inDegree);
    }

} // namespace SPM
