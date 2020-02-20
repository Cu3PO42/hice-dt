#include "split.h"
#include "simple_job_manager.h"
#include "simple_job_manager.impl.h"

using namespace horn_verification;

std::unique_ptr<abstract_job> horn_verification::simple_job_manager::next_job() {
    //
    // Get next slice
    //
    auto sl = _slices.front();
    _slices.pop_front();

    //
    // If this is the first split , split on the unique categorical attribute
    //
    if (_is_first_split) {
        // Check if data points have exactly one categorical attribute
        if (_datapoint_ptrs[sl._left_index]->_categorical_data.size() != 1) {
            throw std::runtime_error("Learner expects exactly one categorical attribute");
        }

        _is_first_split = false;
        return std::make_unique<categorical_split_job>(sl, 0);
    }

    //
    // Determine what needs to be done (split or create leaf)
    //
    bool label(false); // label is unimportant (if is_leaf() returns false)
    std::unordered_set<datapoint<bool> *> positive_ptrs;
    std::unordered_set<datapoint<bool> *> negative_ptrs;

    // Slice can be turned into a leaf node
    if (is_leaf(sl, label, positive_ptrs, negative_ptrs)) {
        return std::make_unique<leaf_creation_job>(
            sl,
            label,
            std::move(positive_ptrs),
            std::move(negative_ptrs));
    }
    // Slice needs to be split
    else {
        return find_best_split(sl);
    }
}

std::unique_ptr<abstract_job> simple_job_manager::find_best_split(const slice &sl) {
    return find_best_split_base<cat_split, int_split, simple_job_manager>(sl);
}

double simple_job_manager::entropy(
    const std::vector<datapoint<bool> *> &datapoint_ptrs, std::size_t left_index, std::size_t right_index) {
    unsigned int count_f = 0;
    unsigned int count_t = 0;

    for (std::size_t i = left_index; i <= right_index; ++i) {
        if (datapoint_ptrs[i]->_is_classified) {
            if (datapoint_ptrs[i]->_classification) {
                ++count_t;
            } else {
                ++count_f;
            }
        }
    }

    double sum = count_t + count_f;
    // std::cout << "sum=" << sum << std::endl;
    double p_t = ((double)(count_t) / sum);
    // std::cout << "p_t=" << p_t << std::endl;
    double p_f = ((double)(count_f) / sum);
    // std::cout << "p_f=" << p_f << std::endl;

    double entropy_t = count_t == 0 ? 0 : p_t * log2(p_t);
    double entropy_f = count_f == 0 ? 0 : p_f * log2(p_f);

    return -(entropy_t + entropy_f);
}

double simple_job_manager::weighted_entropy(
    const std::vector<datapoint<bool> *> &datapoint_ptrs, std::size_t left_index, std::size_t right_index) {
    return entropy(datapoint_ptrs, left_index, right_index) *
           num_classified_points(_datapoint_ptrs, left_index, right_index);
}

unsigned int simple_job_manager::num_classified_points(
    const std::vector<datapoint<bool> *> &datapoint_ptrs, std::size_t left_index, std::size_t right_index) {
    unsigned int count = 0;

    for (std::size_t i = left_index; i <= right_index; ++i) {
        if (datapoint_ptrs[i]->_is_classified) {
            count++;
        }
    }
    return count;
}