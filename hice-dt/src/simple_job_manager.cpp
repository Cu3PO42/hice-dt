#include "simple_job_manager.h"

using namespace horn_verification;

std::unique_ptr<abstract_job> horn_verification::simple_job_manager::next_job() {
    //
    // Get next slice
    //
    auto sl = _slices.front();
    _slices.pop_front();
    // std::cout << sl << std::endl;

    //
    // If this is the first split , split on the unique categorical attribute
    //
    if (_is_first_split) {
        // Check if data points have exactly one categorical attribute
        if (_datapoint_ptrs[sl._left_index]->_categorical_data.size() != 1) {
            throw std::runtime_error("Learner expects exactly one categorical attribute");
        }

        _is_first_split = false;
        return std::unique_ptr<abstract_job>{std::make_unique<categorical_split_job>(sl, 0)};
    }

    //
    // Determine what needs to be done (split or create leaf)
    //
    auto label = false; // label is unimportant (if is_leaf() returns false)
    auto positive_ptrs = std::unordered_set<datapoint<bool> *>();
    auto negative_ptrs = std::unordered_set<datapoint<bool> *>();
    auto can_be_turned_into_leaf = is_leaf(sl, label, positive_ptrs, negative_ptrs);

    // Slice can be turned into a leaf node
    if (can_be_turned_into_leaf) {
        return std::unique_ptr<abstract_job>{
            std::make_unique<leaf_creation_job>(sl, label, std::move(positive_ptrs), std::move(negative_ptrs))};
    }
    // Slice needs to be split
    else {
        return find_best_split(sl);
    }
}

std::unique_ptr<abstract_job> simple_job_manager::find_best_split(const slice &sl) {
    assert(sl._left_index <= sl._right_index && sl._right_index < _datapoint_ptrs.size());

    // 0) Initialize variables
    bool int_split_possible = false;
    double best_int_gain_ratio = 0;
    std::size_t best_int_attribute = 0;
    int best_int_threshold = 0;

    bool cat_split_possible = false;
    double best_cat_gain_ratio = 0;
    std::size_t best_cat_attribute = 0;

    //
    // Process categorical attributes
    //
    for (std::size_t attribute = 0; attribute < _datapoint_ptrs[sl._left_index]->_categorical_data.size();
         ++attribute) {
        // 1) Sort according to categorical attribute
        auto comparer = [attribute](const datapoint<bool> *const a, const datapoint<bool> *const b) {
            return a->_categorical_data[attribute] < b->_categorical_data[attribute];
        };
        std::sort(_datapoint_ptrs.begin() + sl._left_index, _datapoint_ptrs.begin() + sl._right_index + 1, comparer);

        // 2) sum all weighted entropies
        double total_weighted_entropy = 0;
        double total_intrinsic_value = 0;
        bool split_possible = true;
        auto cur_left = sl._left_index;
        auto cur_right = cur_left;

        while (cur_right <= sl._right_index) {
            auto cur_category = _datapoint_ptrs[cur_left]->_categorical_data[attribute];

            while (cur_right + 1 <= sl._right_index &&
                   cur_category == _datapoint_ptrs[cur_right + 1]->_categorical_data[attribute]) {
                ++cur_right;
            }

            // If only one category, skip attribute
            if (cur_left == sl._left_index && cur_right == sl._right_index) {
                split_possible = false;
                break;
            } else {
                total_weighted_entropy += weighted_entropy(_datapoint_ptrs, cur_left, cur_right);

                double n1 = 1.0 * num_classified_points(_datapoint_ptrs, cur_left, cur_right);
                double n = 1.0 * num_classified_points(_datapoint_ptrs, sl._left_index, sl._right_index);
                total_intrinsic_value += (n1 == 0) ? 0.0 : -1.0 * (n1 / n) * log2(n1 / n);

                cur_left = cur_right + 1;
                cur_right = cur_left;
            }
        }
        if (split_possible) {
            double info_gain;
            if (num_classified_points(_datapoint_ptrs, sl._left_index, sl._right_index) == 0) {
                info_gain = entropy(_datapoint_ptrs, sl._left_index, sl._right_index);
            } else {
                info_gain =
                    entropy(_datapoint_ptrs, sl._left_index, sl._right_index) -
                    total_weighted_entropy / num_classified_points(_datapoint_ptrs, sl._left_index, sl._right_index);
            }

            assert(total_intrinsic_value > 0.0);
            double gain_ratio = info_gain / total_intrinsic_value;
            // split is possible on the current "attribute"
            if (!cat_split_possible || gain_ratio > best_cat_gain_ratio) {
                cat_split_possible = true;
                best_cat_gain_ratio = gain_ratio;
                best_cat_attribute = attribute;
            }
        }
    }

    // std::cout << "Now processing integer splits" << std::endl;
    //
    // Process integer attributes
    //
    for (std::size_t attribute = 0; attribute < _datapoint_ptrs[sl._left_index]->_int_data.size(); ++attribute) {
        int tries = 0;
        double best_int_entropy_for_given_attribute = 1000000;
        bool int_split_possible_for_given_attribute = false;
        int best_int_split_index_for_given_attribute = 0;
        double best_intrinsic_value_for_given_attribute = 0;

        // 1) Sort according to int attribute
        auto comparer = [attribute](const datapoint<bool> *const a, const datapoint<bool> *const b) {
            return a->_int_data[attribute] < b->_int_data[attribute];
        };
        std::sort(_datapoint_ptrs.begin() + sl._left_index, _datapoint_ptrs.begin() + sl._right_index + 1, comparer);

        // 2) Try all thresholds of current attribute
        auto cur = sl._left_index;
        while (cur < sl._right_index) {
            // Skip to riight most entry with the same value
            while (cur + 1 <= sl._right_index &&
                   _datapoint_ptrs[cur + 1]->_int_data[attribute] == _datapoint_ptrs[cur]->_int_data[attribute]) {
                ++cur;
            }

            // Split is possible
            if (cur < sl._right_index) {
                tries++;

                // if cuts have been thresholded, check that a split at the current value of the numerical attribute is
                // allowed
                if (!_are_numerical_cuts_thresholded ||
                    ((-1 * _threshold <= _datapoint_ptrs[cur]->_int_data[attribute]) &&
                     (_datapoint_ptrs[cur]->_int_data[attribute] <= _threshold))) {
                    // std::cout << "considering attribute: " << attribute << " sl._left_index: " << cur << " cut: " <<
                    // _datapoint_ptrs[cur]->_int_data[attribute] << std::endl;
                    // weighted_entropy_left = H(left_node) * num_classified_points(left_node)
                    auto weighted_entropy_left = weighted_entropy(_datapoint_ptrs, sl._left_index, cur);
                    // weighted_entropy_right = H(right_node) * num_classified_points(right_node)
                    auto weighted_entropy_right = weighted_entropy(_datapoint_ptrs, cur + 1, sl._right_index);
                    auto total_weighted_entropy = weighted_entropy_left + weighted_entropy_right;

                    if (!int_split_possible_for_given_attribute ||
                        total_weighted_entropy < best_int_entropy_for_given_attribute) {
                        // std::cout << "updated the entropy; split is now definitely possible" << std::endl;
                        int_split_possible_for_given_attribute = true;

                        best_int_entropy_for_given_attribute = total_weighted_entropy;
                        best_int_split_index_for_given_attribute = cur;

                        // computation of the intrinsic value of the attribute
                        double n1 = 1.0 * num_classified_points(_datapoint_ptrs, sl._left_index, cur);
                        double n2 = 1.0 * num_classified_points(_datapoint_ptrs, cur + 1, sl._right_index);
                        double n = n1 + n2;
                        best_intrinsic_value_for_given_attribute = (n1 == 0.0 ? 0.0 : -1.0 * (n1 / n) * log2(n1 / n)) +
                                                                   (n2 == 0.0 ? 0.0 : -1.0 * (n2 / n) * log2(n2 / n));
                    }
                }

                ++cur;
            }
        }
        if (int_split_possible_for_given_attribute) {
            // We have found the best split threshold for the given attribute
            // Now compute the information gain to optimize across different attributes
            double best_info_gain_for_attribute;
            if (num_classified_points(_datapoint_ptrs, sl._left_index, sl._right_index) == 0.0) {
                best_info_gain_for_attribute = entropy(_datapoint_ptrs, sl._left_index, sl._right_index);
            } else {
                best_info_gain_for_attribute =
                    entropy(_datapoint_ptrs, sl._left_index, sl._right_index) -
                    best_int_entropy_for_given_attribute /
                        num_classified_points(_datapoint_ptrs, sl._left_index, sl._right_index);
            }

            double interval = (_datapoint_ptrs[sl._right_index]->_int_data[attribute] -
                               _datapoint_ptrs[sl._left_index]->_int_data[attribute]) /
                              (_datapoint_ptrs[best_int_split_index_for_given_attribute + 1]->_int_data[attribute] -
                               _datapoint_ptrs[best_int_split_index_for_given_attribute]->_int_data[attribute]);

            assert(num_classified_points(_datapoint_ptrs, sl._left_index, sl._right_index) > 0);
            double threshCost = (interval < (double)tries ? log2(interval) : log2(tries)) /
                                num_classified_points(_datapoint_ptrs, sl._left_index, sl._right_index);

            best_info_gain_for_attribute -= threshCost;
            assert(best_intrinsic_value_for_given_attribute > 0.0);
            double best_gain_ratio_for_given_attribute =
                best_info_gain_for_attribute / best_intrinsic_value_for_given_attribute;

            if (!int_split_possible || (best_gain_ratio_for_given_attribute > best_int_gain_ratio) ||
                (best_gain_ratio_for_given_attribute == best_int_gain_ratio &&
                 std::abs(_datapoint_ptrs[best_int_split_index_for_given_attribute]->_int_data[attribute]) <
                     std::abs(best_int_threshold))) {
                // if this is the first attribute for which a split is possible then
                // initialize all variables: best_int_gain_ratio, best_int_attribute, best_int_threshold
                int_split_possible = true;
                best_int_gain_ratio = best_gain_ratio_for_given_attribute;
                best_int_attribute = attribute;
                best_int_threshold = _datapoint_ptrs[best_int_split_index_for_given_attribute]->_int_data[attribute];
            }
        }
    }

    //
    // Return best split
    //

    if (!int_split_possible && !cat_split_possible) {
        throw split_not_possible_error("No split possible!");
    }

    else if (int_split_possible && !cat_split_possible) {
        return std::unique_ptr<abstract_job>{
            std::make_unique<int_split_job>(sl, best_int_attribute, best_int_threshold)};
    }

    else if (!int_split_possible && cat_split_possible) {
        return std::unique_ptr<abstract_job>{std::make_unique<categorical_split_job>(sl, best_cat_attribute)};
    }

    else {
        if (best_int_gain_ratio <= best_cat_gain_ratio) {
            return std::unique_ptr<abstract_job>{
                std::make_unique<int_split_job>(sl, best_int_attribute, best_int_threshold)};
        } else {
            return std::unique_ptr<abstract_job>{std::make_unique<categorical_split_job>(sl, best_cat_attribute)};
        }
    }
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