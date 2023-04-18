#ifndef __KDTREE_H__
#define __KDTREE_H__

#include <vector>
#include <numeric>
#include <algorithm>
#include <exception>
#include <functional>

namespace kdt
{
    /** @brief k-d tree class.
    */
    template <class PointT, typename Scalar = float>
    class KDTree
    {
    public:
        /** @brief The constructors.
        */
        KDTree() : root_(nullptr) {};
        KDTree(const std::vector<PointT>& points) : root_(nullptr) { build(points); }

        /** @brief The destructor.
        */
        ~KDTree() { clear(); }

        /** @brief Re-builds k-d tree.
        */
        void build(const std::vector<PointT>& points)
        {
            using namespace std;
            clear();

            points_ = points;

            vector<size_t> indices(points.size());
            iota(begin(indices), end(indices), 0);

            root_ = buildRecursive(indices.data(), static_cast<int>(points.size()), 0);
        }

        /** @brief Clears k-d tree.
        */
        void clear()
        {
            clearRecursive(root_);
            root_ = nullptr;
            points_.clear();
        }

        /** @brief Validates k-d tree.
        */
        bool validate() const
        {
            try
            {
                validateRecursive(root_, 0);
            }
            catch (const Exception&)
            {
                return false;
            }

            return true;
        }

        void insert(const PointT &point) noexcept
        {
            size_t index = points_.size();
            points_.push_back(point);
            if (!root_) {
                root_ = new Node { index, 0 };
            } else
                insertRecursive(point, root_, index);
        }

        template<typename... Args>
        void emplace(Args && ... args) noexcept
        {
            size_t index = points_.size();
            const auto &point = points_.emplace_back(std::forward<Args>(args)...);
            if (!root_)
                root_ = new Node { index, 0 };
            else
                insertRecursive(point, root_, index);
        }

        /** @brief Searches the nearest neighbor.
        */
        int nnSearch(const PointT& query, Scalar* minDist = nullptr) const
        {
            int guess;
            Scalar _minDist = std::numeric_limits<Scalar>::max();

            nnSearchRecursive(query, root_, &guess, &_minDist);

            if (minDist)
                *minDist = _minDist;

            return guess;
        }

        /** @brief Searches k-nearest neighbors.
        */
        std::vector<int> knnSearch(const PointT& query, int k) const
        {
            KnnQueue queue(k);
            knnSearchRecursive(query, root_, queue, k);

            std::vector<int> indices(queue.size());
            for (size_t i = 0; i < queue.size(); i++)
                indices[i] = queue[i].second;

            return indices;
        }

        /** @brief Searches neighbors within radius.
        */
        std::vector<int> radiusSearch(const PointT& query, Scalar radius) const
        {
            std::vector<int> indices;
            radiusSearchRecursive(query, root_, indices, radius);
            return indices;
        }

    private:

        /** @brief k-d tree node.
        */
        struct Node
        {
            size_t idx;                           //!< index to the original point
            uint axis;                            //!< dimension's axis
            Node* next[2] = { nullptr, nullptr }; //!< pointers to the child nodes
        };

        /** @brief k-d tree exception.
        */
        class Exception : public std::exception { using std::exception::exception; };

        /** @brief Bounded priority queue.
        */
        template <class T, class Compare = std::less<T>>
        class BoundedPriorityQueue
        {
        public:

            BoundedPriorityQueue() = delete;
            BoundedPriorityQueue(size_t bound) : bound_(bound) { elements_.reserve(bound + 1); };

            void push(const T& val)
            {
                auto it = std::find_if(std::begin(elements_), std::end(elements_),
                    [&](const T& element){ return Compare()(val, element); });
                elements_.insert(it, val);

                if (elements_.size() > bound_)
                    elements_.resize(bound_);
            }

            const T& back() const { return elements_.back(); };
            const T& operator[](size_t index) const { return elements_[index]; }
            size_t size() const { return elements_.size(); }

        private:
            size_t bound_;
            std::vector<T> elements_;
        };

        /** @brief Priority queue of <distance, index> pair.
        */
        using KnnQueue = BoundedPriorityQueue<std::pair<Scalar, int>>;

        /** @brief Builds k-d tree recursively.
        */
        Node* buildRecursive(size_t* indices, int npoints, uint depth)
        {
            if (npoints <= 0)
                return nullptr;

            const uint axis = depth % PointT::DIM;
            const int mid = (npoints - 1) / 2;

            std::nth_element(indices, indices + mid, indices + npoints, [&](int lhs, int rhs)
            {
                return points_[lhs][axis] < points_[rhs][axis];
            });

            return new Node{
                indices[mid], axis,
                buildRecursive(indices, mid, depth + 1),
                buildRecursive(indices + mid + 1, npoints - mid - 1, depth + 1)
            };
        }

        /** @brief Search k-d tree recursively and insert point.
        */
        void insertRecursive(const PointT &point, Node *node, size_t pointIndex) noexcept
        {
            const PointT &current = points_[node->idx];
            int idx = static_cast<int>(current[node->axis] > point[node->axis]);
            Node *next = node->next[idx];
            if (next)
                insertRecursive(point, next, pointIndex);
            else
                node->next[idx] = new Node{ pointIndex, (node->axis + 1) % PointT::DIM };
        }

        /** @brief Clears k-d tree recursively.
        */
        void clearRecursive(Node* node)
        {
            if (node == nullptr)
                return;

            if (node->next[0])
                clearRecursive(node->next[0]);

            if (node->next[1])
                clearRecursive(node->next[1]);

            delete node;
        }

        /** @brief Validates k-d tree recursively.
        */
        void validateRecursive(const Node* node, int depth) const
        {
            if (node == nullptr)
                return;

            const int axis = node->axis;
            const Node* node0 = node->next[0];
            const Node* node1 = node->next[1];

            if (node0 && node1)
            {
                if (points_[node->idx][axis] < points_[node0->idx][axis])
                    throw Exception();

                if (points_[node->idx][axis] > points_[node1->idx][axis])
                    throw Exception();
            }

            if (node0)
                validateRecursive(node0, depth + 1);

            if (node1)
                validateRecursive(node1, depth + 1);
        }

        /** @brief Searches the nearest neighbor recursively.
        */
        void nnSearchRecursive(const PointT& query, const Node* node, int *guess, Scalar *minDist) const
        {
            if (node == nullptr)
                return;

            const PointT& train = points_[node->idx];

            const Scalar dist = query.distance(train);
            if (dist < *minDist)
            {
                *minDist = dist;
                *guess = node->idx;
            }

            const int axis = node->axis;
            const int dir = query[axis] < train[axis] ? 0 : 1;
            nnSearchRecursive(query, node->next[dir], guess, minDist);

            const Scalar diff = fabs(query[axis] - train[axis]);
            if (diff < *minDist)
                nnSearchRecursive(query, node->next[!dir], guess, minDist);
        }

        /** @brief Searches k-nearest neighbors recursively.
        */
        void knnSearchRecursive(const PointT& query, const Node* node, KnnQueue& queue, int k) const
        {
            if (node == nullptr)
                return;

            const PointT& train = points_[node->idx];

            const Scalar dist = query.distance(train);
            queue.push(std::make_pair(dist, node->idx));

            const int axis = node->axis;
            const int dir = query[axis] < train[axis] ? 0 : 1;
            knnSearchRecursive(query, node->next[dir], queue, k);

            const Scalar diff = fabs(query[axis] - train[axis]);
            if ((int)queue.size() < k || diff < queue.back().first)
                knnSearchRecursive(query, node->next[!dir], queue, k);
        }

        /** @brief Searches neighbors within radius.
        */
        void radiusSearchRecursive(const PointT& query, const Node* node, std::vector<int>& indices, Scalar radius) const
        {
            if (node == nullptr)
                return;

            const PointT& train = points_[node->idx];

            const Scalar dist = query.distance(train);
            if (dist < radius)
                indices.push_back(node->idx);

            const int axis = node->axis;
            const int dir = query[axis] < train[axis] ? 0 : 1;
            radiusSearchRecursive(query, node->next[dir], indices, radius);

            const Scalar diff = fabs(query[axis] - train[axis]);
            if (diff < radius)
                radiusSearchRecursive(query, node->next[!dir], indices, radius);
        }

        Node* root_;                 //!< root node
        std::vector<PointT> points_; //!< points
    };
} // kdt

#endif // !__KDTREE_H__
