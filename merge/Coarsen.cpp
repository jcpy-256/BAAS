

#include "Coarsen.hpp"
// #include "DAG.hpp"

using namespace SPM;

namespace Coarsen
{

    // 检测以node为父节点，其子节点和node是否构成forward tree
    bool isSingleForwardTree(int node, int &nchild, std::vector<int> &child_node, std::vector<bool> &visited, const std::vector<int> &DAG_ptr, const std::vector<int> &DAG_set, const std::vector<int> &DAG_inv_ptr, const std::vector<int> &DAG_inv_set)
    {
        child_node.clear();
        bool flag = true;
        for (int child_ptr = DAG_ptr[node] + 1; child_ptr < DAG_ptr[node + 1]; child_ptr++)
        {
            int child = DAG_set[child_ptr];
            // if (visited[child] == false && child != node)
            if (visited[child] == false) // CSC ascending rowidx
            {
                nchild++;
                visited[child] = true;
                child_node.push_back(child);
                if (DAG_inv_ptr[child + 1] - DAG_inv_ptr[child] > 2)
                {
                    flag = false;
                }
                else if (DAG_inv_ptr[child + 1] - DAG_inv_ptr[child] == 2)
                {
                }
                else
                {
                    flag = false;
                    std::cerr << "Invaliad DAG in CSC format" << std::endl;
                }
            }
        }
        return flag;
    }

    // 检测以node为父节点，其父节点和node是否构成reverse tree
    bool isSingleReverseTree(int node, int &nparent, std::vector<int> &parent_node, std::vector<bool> &visited, const std::vector<int> &DAG_ptr, const std::vector<int> &DAG_set, const std::vector<int> &DAG_inv_ptr, const std::vector<int> &DAG_inv_set)
    {
        parent_node.clear();
        bool flag = true;
        for (int parent_ptr = DAG_inv_ptr[node]; parent_ptr < DAG_inv_ptr[node + 1] - 1; parent_ptr++)
        {
            int parent = DAG_inv_set[parent_ptr];
            // if (visited[parent] == false && parent != node) // 非本节点，并且未被访问。
            if (visited[parent] == false) // CSR colidx ascending
            {
                visited[parent] = true;
                nparent++;
                parent_node.push_back(parent); // 记录子节点，根据flag判断是加入group_stack还是root队列
                if (DAG_ptr[parent + 1] - DAG_ptr[parent] > 2)
                {
                    flag = false;
                }
                else if (DAG_ptr[parent + 1] - DAG_ptr[parent] == 2)
                {
                }
                else
                {
                    flag = false;
                    std::cerr << "Invaliad DAG in CSC format" << std::endl;
                }
            }
        }
        return flag;
    }

    bool forwardTreeCoarseningBFS_all(const int n, const std::vector<int> &DAG_ptr, const std::vector<int> &DAG_set,
                                      int &ngroups, std::vector<int> &group_ptr, std::vector<int> &group_set, bool restriction /**false */)
    {
        // printf("Forward Coarsening all.\n");
        int size_restriction = n;
        int edge_counter = DAG_ptr[n]; // the edge number of the DAG

        // Create tree grouping
        // std::vector<int> group_DAG_ptr;
        // std::vector<int> group_DAG_set;

        // Create the inverse DAG
        DAG dag(n, edge_counter, DAG_set.data(), DAG_ptr.data(), DAG_MAT::DAG_CSC);

        DAG *DAG_inv = DAG::inverseDAG(dag);
        std::vector<int> &DAG_inv_ptr = DAG_inv->DAG_ptr;
        std::vector<int> &DAG_inv_set = DAG_inv->DAG_set;

        // Find the sink node of the inverse DAG (source node of the DAG)
        std::list<int> root_id;
        for (int i = 0; i < n; i++)
        {
            if (DAG_inv_ptr[i + 1] - DAG_inv_ptr[i] == 1)
            {
                root_id.push_back(i);
            }
        }

        // Apply a traversal to this DAG and group sub-trees
        // All the nodes in each group should have one outgoing edge in inverse DAG except the sink node
        std::vector<bool> visited(n, false);
        std::list<int> group_stack;

        if (!group_ptr.empty())
        {
            group_ptr.clear();
        }
        if (!group_set.empty())
        {
            group_set.clear();
        }

        group_set.resize(n);
        group_ptr.reserve(n);

        int nchild = 0;
        std::vector<int> children;
        children.reserve(n);

        int set_cnt = 0;
        ngroups = 0;
        while (!root_id.empty())
        {
            // First visit, first save
            auto head = root_id.front();
            root_id.pop_front();
            visited[head] = true;
            group_stack.push_back(head);
            group_ptr.push_back(set_cnt);
            int lastsize = set_cnt;
            ngroups++;
            while (!group_stack.empty())
            {
                head = group_stack.front();
                group_stack.pop_front();
                int finish_group = false;
                group_set[set_cnt++] = head;

                nchild = 0;
                // check the child is whether satisfy the forward sub-tree condition
                bool isTree = isSingleForwardTree(head, nchild, children, visited, DAG_ptr, DAG_set, DAG_inv_ptr, DAG_inv_set);
                if (isTree)
                {
                    // copy child node to stack
                    group_stack.insert(group_stack.end(), children.begin(), children.begin() + nchild);
                }
                else
                {
                    root_id.insert(root_id.end(), children.begin(), children.begin() + nchild);
                }
            }
        }

        if (group_ptr.back() != set_cnt)
        {
            group_ptr.push_back(set_cnt);
        }
        // printf("set cnt: %d\n", set_cnt);

#ifndef NDEBUG
        std::vector<bool> marks(n, false);
        for (auto &iter : group_set)
        {
            marks[iter] = true;
        }

        for (auto &&mark : marks)
        {
            assert(mark == true);
        }

#endif
        std::vector<int> group_id(n, 0); // node index -> the minmum node idx of a group
        std::vector<int> group_size(n + 1, 0);

#pragma omp parallel for
        for (int g = 0; g < ngroups; g++)
        {
            int id = n; // bind the group to the minmum node of this group that is the root of forward tree
            for (int i = group_ptr[g]; i < group_ptr[g + 1]; i++)
            {
                int node = group_set[i];
                if (node < id)
                {
                    id = node;
                }
            }
            assert(id == group_set[group_ptr[g]]);
            group_size[id + 1] = group_ptr[g + 1] - group_ptr[g];
            for (int i = group_ptr[g]; i < group_ptr[g + 1]; i++)
            {
                int node = group_set[i];
                group_id[node] = id;
            }
        }

        // Sort the nodes based on their ids using counting sort
        for (int i = 1; i < n; i++)
        {
            group_size[i + 1] += group_size[i];
        }
        group_ptr.clear();
        group_ptr.push_back(0);

        for (int i = 0; i < n; i++)
        {
            if (group_size[i + 1] != group_size[i])
            {
                group_ptr.push_back(group_size[i + 1]);
            }
        }

        std::vector<int> unordered_group_set(n);
// #pragma omp parallel for
        for (int i = 0; i < n; i++)
        {
            unordered_group_set[group_size[group_id[group_set[i]]]++] = group_set[i];
            // __sync_fetch_and_add(&group_size[group_id[group_set[i]]], 1);
            // group_size[group_id[group_set[i]]]++
        }

        // group_set = unordered_group_set;
        group_set = std::move(unordered_group_set);
        assert(group_set.size() == group_ptr.back());
        delete DAG_inv;
        return true;
    }

    /**
     * 这个函数不会处理得到完整的reverse tree而是将出现group_stack中的顶点不满足条件就立刻删除了group_stack中的其他节点，限制了粗化的条件。
     */
    bool reverseTreeCoarseningBFS_all(const int n, const std::vector<int> &DAG_ptr, const std::vector<int> &DAG_set, int &ngroups, std::vector<int> &group_ptr, std::vector<int> &group_set, bool restriction /**false */)
    {
        // printf("Reverse Coarsening all.\n");

        int size_restriction = n;

        int edge_counter = DAG_ptr[n]; // the edge number of the DAG

        // Create tree grouping
        // std::vector<int> group_DAG_ptr;
        // std::vector<int> group_DAG_set;

        // Create inverse DAG : transpose csc to csr
        DAG dag(n, edge_counter, DAG_set.data(), DAG_ptr.data(), DAG_MAT::DAG_CSC);

        DAG *DAG_inv = DAG::inverseDAG(dag);

        std::vector<int> &DAG_inv_ptr = DAG_inv->DAG_ptr;
        std::vector<int> &DAG_inv_set = DAG_inv->DAG_set;

        // Find the leaves of the original DAG (root node of inverse DAG)
        // that is the nodes without any children depending on it
        std::list<int> root_id;
        for (int i = 0; i < n; i++)
        {
            if (DAG_ptr[i + 1] - DAG_ptr[i] == 1)
            {
                root_id.push_back(i);
            }
        }

        // Apply a traversal to this DAG and group reverse sub-tree
        // All the nodes in each group should have outgoing edge of one except the root
        std::vector<bool> visited(n, false); // visited flag
        std::list<int> group_stack;          // store the node of a reverse sub-tree

        if (!group_ptr.empty())
        {
            group_ptr.clear();
        }

        if (!group_set.empty())
        {
            group_set.clear();
        }

        group_set.resize(n);
        group_ptr.reserve(n);

        int nparent = 0;
        std::vector<int> parents;
        parents.reserve(n);

        int set_cnt = 0;
        ngroups = 0;

        // The BFS Traversal
        while (!root_id.empty())
        {
            // First visit. first serve
            auto head = root_id.front();
            root_id.pop_front();
            visited[head] = true;
            group_stack.push_back(head);
            group_ptr.push_back(set_cnt);
            int last_size = set_cnt;
            ngroups++;
            while (!group_stack.empty())
            {
                head = group_stack.front();
                group_stack.pop_front();

                int finish_group = false;
                group_set[set_cnt++] = head;
                nparent = 0;

                // check the parent is whether satisfy the reverse sub-tree condition
                bool isTree = isSingleReverseTree(head, nparent, parents, visited, DAG_ptr, DAG_set, DAG_inv_ptr, DAG_inv_set);
                if (isTree)
                {
                    group_stack.insert(group_stack.end(), parents.begin(), parents.end());
                }
                else
                {
                    root_id.insert(root_id.end(), parents.begin(), parents.end());
                }
            }
        }

        if (group_ptr.back() != set_cnt)
        {
            group_ptr.push_back(set_cnt);
        }
#ifndef NDEBUG
        std::vector<bool> marks(n, false);
        for (auto &iter : group_set)
        {
            marks[iter] = true;
        }

        for (auto &&item : visited)
        {
            assert(item == true);
        }

        for (auto &&mark : marks)
        {
            assert(mark == true);
        }
#endif
        std::vector<int> group_id(n, 0);       // Note: the small id in a group set,the vector is a mapping: index-> value (nodeidex -> groupid)
        std::vector<int> group_size(n + 1, 0); // Note: the node num of a group set\

// NOTE: We have to do these stuff so that the resulted DAG will be a lower triangular one
// Sort Groups and give each group a name based on the minimum of the ids inside a group
#pragma omp parallel for schedule(static)
        for (int g = 0; g < ngroups; g++)
        {
            // maximum id
            int id = 0; // bind this group to the maximum node that is the root node of reverse tree
            for (int i = group_ptr[g]; i < group_ptr[g + 1]; i++)
            {
                int node = group_set[i];
                if (node > id)
                {
                    id = node;
                }
            }
            assert(id == group_set[group_ptr[g]]); // ensure the maximum node is the root of reverse tree
            group_size[id + 1] = group_ptr[g + 1] - group_ptr[g];

            for (int i = group_ptr[g]; i < group_ptr[g + 1]; i++)
            {
                int node = group_set[i];
                group_id[node] = id;
            }
        }

        for (int i = 1; i < n; i++)
        {
            group_size[i + 1] += group_size[i];
        }

        group_ptr.clear();
        group_ptr.push_back(0);

        for (int i = 0; i < n; i++)
        {
            if (group_size[i + 1] != group_size[i])
            {
                group_ptr.push_back(group_size[i + 1]);
            }
        }
        // assert(group_size.back() == dag.getNodeNum);

        // sort the node in a group by original order
        std::vector<int> unordered_group_set(n);
// #pragma omp parallel for
        for (int i = 0; i < n; i++)
        {
            unordered_group_set[group_size[group_id[group_set[i]]]++] = group_set[i];
            // __sync_fetch_and_add(&group_size[group_id[group_set[i]]], 1);
        }

        group_set = std::move(unordered_group_set);
        // printf("stake\n");
        delete DAG_inv;
        // fflush(stdout);
        return true;
    }

    bool forwardTreeCoarseningBFS(int n, std::vector<int> &DAG_ptr, std::vector<int> &DAG_set,
                                  int &ngroups, std::vector<int> &group_ptr, std::vector<int> &group_set, bool restriction /**false */)
    {
        printf("Forward Coarsening part.\n");
        int size_restriction = n;
        int edge_counter = DAG_ptr[n]; // the edge number of the DAG

        // Create tree grouping
        std::vector<int> group_DAG_ptr;
        std::vector<int> group_DAG_set;

        // Create the inverse DAG
        DAG dag(n, edge_counter, DAG_set.data(), DAG_ptr.data(), DAG_MAT::DAG_CSC);

        DAG *DAG_inv = DAG::inverseDAG(dag);
        std::vector<int> &DAG_inv_ptr = DAG_inv->DAG_ptr;
        std::vector<int> &DAG_inv_set = DAG_inv->DAG_set;
        delete DAG_inv;

        // Find the sink node of the inverse DAG (source node of the DAG)
        std::list<int> root_id;
        for (int i = 0; i < n; i++)
        {
            if (DAG_inv_ptr[i + 1] - DAG_inv_ptr[i] == 1)
            {
                root_id.push_back(i);
            }
        }

        // Apply a traversal to this DAG and group sub-trees
        // All the nodes in each group should have one outgoing edge in inverse DAG except the sink node
        std::vector<bool> visited(n, false);
        std::list<int> group_stack;

        if (!group_ptr.empty())
        {
            group_ptr.clear();
        }
        if (!group_set.empty())
        {
            group_set.clear();
        }

        group_set.resize(n);
        group_ptr.reserve(n);

        int set_cnt = 0;
        ngroups = 0;
        while (!root_id.empty())
        {
            // First visit, first save
            auto head = root_id.front();
            root_id.pop_front();
            visited[head] = true;
            group_stack.push_back(head);
            group_ptr.push_back(set_cnt);
            int lastsize = set_cnt;
            ngroups++;
            while (!group_stack.empty())
            {
                head = group_stack.front();
                group_stack.pop_front();
                int finish_group = false;
                group_set[set_cnt++] = head;
                for (int child_ptr = DAG_ptr[head] + 1; child_ptr < DAG_ptr[head + 1]; child_ptr++)
                {
                    int child = DAG_set[child_ptr];
                    if (!visited[child])
                    {
                        visited[child] = true;
                        if (DAG_inv_ptr[child + 1] - DAG_inv_ptr[child] > 2)
                        {
                            root_id.push_back(child);
                            finish_group = true;
                        }
                        else if (DAG_inv_ptr[child + 1] - DAG_inv_ptr[child] == 2)
                        {
                            group_stack.push_back(child);
                        }
                        else
                        {
                            std::cerr << "Invalid DAG in CSC format" << std::endl;
                        }
                    }
                }
                if (finish_group == true || set_cnt - lastsize == size_restriction)
                {
                    while (!group_stack.empty())
                    {
                        auto tmp = group_stack.front();
                        group_stack.pop_front();
                        root_id.push_back(tmp);
                    }
                }
            }
        }

        if (group_ptr.back() != set_cnt)
        {
            group_ptr.push_back(set_cnt);
        }
        // printf("set cnt: %d\n", set_cnt);

#ifndef NDEBUG
        std::vector<bool> marks(n, false);
        for (auto &iter : group_set)
        {
            marks[iter] = true;
        }

        for (auto &&mark : marks)
        {
            assert(mark == true);
        }

#endif
        std::vector<int> group_id(n, 0); // node index -> the minmum node idx of a group
        std::vector<int> group_size(n + 1, 0);

        for (int g = 0; g < ngroups; g++)
        {
            int id = n;
            for (int i = group_ptr[g]; i < group_ptr[g + 1]; i++)
            {
                int node = group_set[i];
                if (node < id)
                {
                    id = node;
                }
            }
            group_size[id + 1] = group_ptr[g + 1] - group_ptr[g];
            for (int i = group_ptr[g]; i < group_ptr[g + 1]; i++)
            {
                int node = group_set[i];
                group_id[node] = id;
            }
        }

        // Sort the nodes based on their ids using counting sort
        for (int i = 1; i < n; i++)
        {
            group_size[i + 1] += group_size[i];
        }
        group_ptr.clear();
        group_ptr.push_back(0);

        for (int i = 0; i < n; i++)
        {
            if (group_size[i + 1] != group_size[i])
            {
                group_ptr.push_back(group_size[i + 1]);
            }
        }

        std::vector<int> unordered_group_set(n);
        for (int i = 0; i < n; i++)
        {
            unordered_group_set[group_size[group_id[group_set[i]]]++] = group_set[i];
        }

        group_set = unordered_group_set;
        return true;
    }

    /**
     * 这个函数不会处理得到完整的reverse tree而是将出现group_stack中的顶点不满足条件就立刻删除了group_stack中的其他节点，限制了粗化的条件。
     */
    bool reverseTreeCoarseningBFS(int n, std::vector<int> &DAG_ptr, std::vector<int> &DAG_set, int &ngroups, std::vector<int> &group_ptr, std::vector<int> &group_set, bool restriction /**false */)
    {

        printf("Reverse Coarsening part.\n");

        int size_restriction = n;

        int edge_counter = DAG_ptr[n]; // the edge number of the DAG

        // Create tree grouping
        std::vector<int> group_DAG_ptr;
        std::vector<int> group_DAG_set;

        // Create inverse DAG : transpose csc to csr
        DAG dag(n, edge_counter, DAG_set.data(), DAG_ptr.data(), DAG_MAT::DAG_CSC);

        DAG *DAG_inv = DAG::inverseDAG(dag);

        std::vector<int> DAG_inv_ptr = DAG_inv->DAG_ptr;
        std::vector<int> DAG_inv_set = DAG_inv->DAG_set;
        delete DAG_inv;

        // Find the leaves of the original DAG (root node of inverse DAG)
        // that is the nodes without any children depending on it
        std::list<int> root_id;
        for (int i = 0; i < n; i++)
        {
            if (DAG_ptr[i + 1] - DAG_ptr[i] == 1)
            {
                root_id.push_back(i);
            }
        }

        // Apply a traversal to this DAG and group reverse sub-tree
        // All the nodes in each group should have outgoing edge of one except the root
        std::vector<bool> visited(n, false); // visited flag
        std::list<int> group_stack;          // store the node of a reverse sub-tree

        if (!group_ptr.empty())
        {
            group_ptr.clear();
        }

        if (group_set.empty())
        {
            group_set.clear();
        }

        group_set.resize(n);
        group_ptr.reserve(n);

        int set_cnt = 0;
        ngroups = 0;

        // The BFS Traversal
        while (!root_id.empty())
        {

            // First visit. first serve
            auto head = root_id.front();
            root_id.pop_front();
            visited[head] = true;
            group_stack.push_back(head);
            group_ptr.push_back(set_cnt);
            int last_size = set_cnt;
            ngroups++;
            while (!group_stack.empty())
            {
                head = group_stack.front();
                group_stack.pop_front();

                int finish_group = false;
                group_set[set_cnt++] = head;
                // check the node parent and the parent's children
                for (int parent_ptr = DAG_inv_ptr[head]; parent_ptr < DAG_inv_ptr[head + 1] - 1; parent_ptr++)
                {
                    int parent = DAG_inv_set[parent_ptr];
                    if (!visited[parent])
                    {
                        visited[parent] = true;
                        if (DAG_ptr[parent + 1] - DAG_ptr[parent] > 2)
                        {
                            root_id.push_back(parent);
                            finish_group = true;
                        }
                        else if (DAG_ptr[parent + 1] - DAG_ptr[parent] == 2) // only has one child that is head
                        {
                            group_stack.push_back(parent);
                        }
                        else
                        {
                            std::cerr << "Invalid DAG in CSC format" << std::endl;
                        }
                    }
                }
                if (finish_group == true || set_cnt - last_size == size_restriction)
                {
                    while (!group_stack.empty())
                    {
                        auto tmp = group_stack.front();
                        group_stack.pop_front();
                        root_id.push_back(tmp);
                    }
                }
            }
        }

        if (group_stack.back() != set_cnt)
        {
            group_ptr.push_back(set_cnt);
        }
#ifndef NDEBUG
        std::vector<bool> marks(n, false);
        for (auto &iter : group_set)
        {
            marks[iter] = true;
        }

        for (auto &&mark : marks)
        {
            assert(mark == true);
        }
#endif
        std::vector<int> group_id(n, 0);       // Note: the small id in a group set,the vector is a mapping: index-> value (nodeidex -> groupid)
        std::vector<int> group_size(n + 1, 0); // Note: the node num of a group set\

        // NOTE: We have to do these stuff so that the resulted DAG will be a lower triangular one
        // Sort Groups and give each group a name based on the minimum of the ids inside a group

        for (int g = 0; g < ngroups; g++)
        {
            // smallest id
            int id = n;
            for (int i = group_ptr[g]; i < group_ptr[g + 1]; i++)
            {
                int node = group_set[i];
                if (node > id)
                {
                    id = node;
                }
            }
            group_size[id + 1] = group_ptr[g + 1] - group_ptr[g];

            for (int i = group_ptr[g]; i < group_ptr[g + 1]; i++)
            {
                int node = group_set[i];
                group_id[node] = id;
            }
        }

        for (int i = 1; i < n; i++)
        {
            group_size[i + 1] += group_size[i];
        }

        group_ptr.clear();
        group_ptr.push_back(0);

        for (int i = 0; i < n; i++)
        {
            if (group_size[i + 1] != group_size[i])
            {
                group_ptr.push_back(group_size[i + 1]);
            }
        }
        // sort the node in a group by original order
        std::vector<int> unordered_group_set(n);
        for (int i = 0; i < n; i++)
        {
            unordered_group_set[group_size[group_id[group_set[i]]]++] = group_set[i];
        }

        group_set = unordered_group_set;
        return true;
    }

    void buildGroupDAGParallel(const int &n, const int &ngroups, const int *group_ptr, const int *group_set,
                               const int *DAG_ptr, const int *DAG_set, std::vector<int> &group_DAG_ptr, std::vector<int> &group_DAG_set)
    {
        std::vector<std::vector<int>> DAG(ngroups);
        // Computing group inverse
        std::vector<int> group_inv(n); // Note:
        int nnz = 0;

#pragma omp parallel
        {
#pragma omp for
            for (int g = 0; g < ngroups; g++)
            {
                for (int i = group_ptr[g]; i < group_ptr[g + 1]; i++)
                {
                    int node = group_set[i];
                    group_inv[node] = g; // node idx-> group idx
                }
            }

#pragma omp for reduction(+ : nnz)
            for (int g = 0; g < ngroups; g++)
            {
                for (int node_ptr = group_ptr[g]; node_ptr < group_ptr[g + 1]; node_ptr++)
                {
                    int node = group_set[node_ptr];
                    for (int child_ptr = DAG_ptr[node]; child_ptr < DAG_ptr[node + 1]; child_ptr++) // Note: add edges to group
                    {
                        int child = group_inv[DAG_set[child_ptr]]; // the group of its children
                        DAG[g].push_back(child);
                        assert(child >= g);
                    }
                }
                // sort and  delete duplicate edge
                std::sort(DAG[g].begin(), DAG[g].end());
                DAG[g].erase(std::unique(DAG[g].begin(), DAG[g].end()), DAG[g].end());
                nnz += DAG[g].size();
            }
        }

        group_DAG_ptr.resize(ngroups + 1, 0);
        group_DAG_set.resize(nnz);
        // add edges to grouped DAG CSC
        long int cti, edges = 0;
        for (cti = 0, edges = 0; cti < ngroups; cti++)
        {
            group_DAG_ptr[cti] = edges;
            for (int ctj = 0; ctj < DAG[cti].size(); ctj++)
            {
                group_DAG_set[edges++] = DAG[cti][ctj];
            }
        }
        group_DAG_ptr[cti] = edges;
    }

    void buildGroupDAG(const int &n, const int &ngroups, const int *group_ptr, const int *group_set,
                       const int *DAG_ptr, const int *DAG_set, std::vector<int> &group_DAG_ptr, std::vector<int> &group_DAG_set)
    {
        // Computing group inverse
        std::vector<int> group_inv(n); // Note:
#pragma omp parallel for schedule(static)
        for (int g = 0; g < ngroups; g++)
        {
            for (int i = group_ptr[g]; i < group_ptr[g + 1]; i++)
            {
                int node = group_set[i];
                group_inv[node] = g; // node idx-> group idx
            }
        }

        std::vector<std::vector<int>> DAG(ngroups);
        int nnz = 0;
#pragma omp parallel for reduction(+ : nnz)
        for (int g = 0; g < ngroups; g++)
        {
            for (int node_ptr = group_ptr[g]; node_ptr < group_ptr[g + 1]; node_ptr++)
            {
                int node = group_set[node_ptr];
                for (int child_ptr = DAG_ptr[node]; child_ptr < DAG_ptr[node + 1]; child_ptr++) // Note: add edges to group
                {
                    int child = group_inv[DAG_set[child_ptr]]; // the group of its children
                    DAG[g].push_back(child);
                    assert(child >= g);
                }
            }
            // sort and  delete duplicate edge
            std::sort(DAG[g].begin(), DAG[g].end());
            DAG[g].erase(std::unique(DAG[g].begin(), DAG[g].end()), DAG[g].end());
            nnz += DAG[g].size();
        }

        group_DAG_ptr.resize(ngroups + 1, 0);
        group_DAG_set.resize(nnz);
        // add edges to grouped DAG CSC
        long int cti, edges = 0;
        for (cti = 0, edges = 0; cti < ngroups; cti++)
        {
            group_DAG_ptr[cti] = edges;
            for (int ctj = 0; ctj < DAG[cti].size(); ctj++)
            {
                group_DAG_set[edges++] = DAG[cti][ctj];
            }
        }
        group_DAG_ptr[cti] = edges;
    }

    void costComputation(int nodes, const int *colptr, int *rowidx, const int *rowptr, const int *colidx,
                         Kernel kernel, const int *group_ptr, const int *group_set, bool grouped, std::vector<double> &cost)
    {
        // if (!cost.empty())
        // {
        cost.clear();
        cost.resize(nodes, 0);
        // }

        if (kernel == SpTRSV_LL)
        {
            const int *Lp = rowptr;
            if (grouped)
            {
#pragma omp parallel for
                for (int g = 0; g < nodes; g++)
                {
                    for (int node_ptr = group_ptr[g]; node_ptr < group_ptr[g + 1]; node_ptr++)
                    {
                        int node = group_set[node_ptr];
                        cost[g] += 1 * (Lp[node + 1] - Lp[node]);
                    }
                }
            }
            else
            {
#pragma omp parallel for
                for (int row = 0; row < nodes; row++)
                {
                    cost[row] += Lp[row + 1] - Lp[row];
                }
            }
        }
        else
        {
            std::wcerr << "The kernel is not supported, using cost 1 for each node" << std::endl;
        }
    }

    /**
     * mapping a group-group to node. the group_ptr_f is based on group_ptr, we must merge it to get a new group
     */
    void groupRemapping(std::vector<int> &group_ptr, std::vector<int> &group_set, const std::vector<int> group_ptr_f, const std::vector<int> group_set_f, const int ngroups_f)
    {
        std::vector<int> group_ptr_new;
        std::vector<int> group_set_new;

        std::vector<int> ptr;
        std::vector<int> set;
        group_ptr_new.reserve(ngroups_f + 1);
        group_set_new.reserve(group_set.size()); // the size equals to node number

        int offset = 0;
        group_ptr_new.push_back(0);
        for (int g_f = 0; g_f < ngroups_f; g_f++) // traversal: group_ptr_f
        {
            for (int g_ptr = group_ptr_f[g_f]; g_ptr < group_ptr_f[g_f + 1]; g_ptr++) // traverse: group_set_f
            {
                assert(g_ptr < group_set_f.size());
                int g = group_set_f[g_ptr];
                assert(g < group_ptr.size());
                int beg = group_ptr[g];
                int end = group_ptr[g + 1];
                // copy data
                offset += end - beg;
                group_set_new.insert(group_set_new.end(), group_set.begin() + beg, group_set.begin() + end);
            }
            group_ptr_new.push_back(offset);
        }
        assert(group_ptr_new.size() == ngroups_f + 1);
        assert(group_set_new.size() == group_ptr_new.back());

        group_ptr = std::move(group_ptr_new);
        group_set = std::move(group_set_new);
    }

} // namespace Coarsen
