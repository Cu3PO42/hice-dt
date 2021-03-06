#include "complex_job_manager.h"

using namespace horn_verification;

void complex_job_manager::initialize_datapoint_ptrs_to_frac() {
    for (auto & _datapoint_ptr : _datapoint_ptrs) {
        if (_datapoint_ptr->_is_classified) {
            if (_datapoint_ptr->_classification) {
                _datapoint_ptrs_to_frac[_datapoint_ptr] = 1.0;
            } else {
                _datapoint_ptrs_to_frac[_datapoint_ptr] = 0.0;
            }
        }
    }
}

void complex_job_manager::update_datapoint_ptrs_to_frac_with_complete_horn_assignments() {
    std::map<datapoint<bool> *, double> _sum_of_datapoint_ptrs_to_frac;

    // Create a list of _datapoint_ptrs which are unclassified
    std::vector<datapoint<bool> *> _unclassified_datapoint_ptrs_stable;
    for (auto & _datapoint_ptr : _datapoint_ptrs) {
        if (!_datapoint_ptr->_is_classified) {
            _unclassified_datapoint_ptrs_stable.push_back(_datapoint_ptr);
        }
    }

    // Now generate complete horn assignments by randomly picking a datapoint from _unclassified_datapoint_ptrs and
    // assigning it True/False followed by label propagation.
    int numberOfCompleteHornAssignments = 3;
    for (int i = 0; i < numberOfCompleteHornAssignments; i++) {
        std::vector<datapoint<bool> *> _unclassified_datapoint_ptrs_temp(_unclassified_datapoint_ptrs_stable);
        auto positive_ptrs = std::unordered_set<datapoint<bool> *>();
        auto negative_ptrs = std::unordered_set<datapoint<bool> *>();
        while (!_unclassified_datapoint_ptrs_temp.empty()) {
            unsigned int itemToAssignClassification = rand() % _unclassified_datapoint_ptrs_temp.size();
            if (rand() % 2 == 0) {
                positive_ptrs.insert(_unclassified_datapoint_ptrs_temp[itemToAssignClassification]);
            } else {
                negative_ptrs.insert(_unclassified_datapoint_ptrs_temp[itemToAssignClassification]);
            }

            auto ok = _horn_solver.solve(_datapoint_ptrs, _horn_constraints, positive_ptrs, negative_ptrs);
            assert(ok);

            // Remove the items present in positive_ptrs and negative_ptrs from _unclassified_datapoint_ptrs_temp
            for (auto positive_ptr : positive_ptrs) {
                // If *it is present in _unclassified_datapoint_ptrs_temp, then remove it.
                auto result_it = std::find(
                    _unclassified_datapoint_ptrs_temp.begin(), _unclassified_datapoint_ptrs_temp.end(), positive_ptr);
                if (result_it != _unclassified_datapoint_ptrs_temp.end()) {
                    _unclassified_datapoint_ptrs_temp.erase(result_it);
                }
            }
            for (auto negative_ptr : negative_ptrs) {
                // If *it is present in _unclassified_datapoint_ptrs_temp, then remove it.
                auto result_it = std::find(
                    _unclassified_datapoint_ptrs_temp.begin(), _unclassified_datapoint_ptrs_temp.end(), negative_ptr);
                if (result_it != _unclassified_datapoint_ptrs_temp.end()) {
                    _unclassified_datapoint_ptrs_temp.erase(result_it);
                }
            }
        }
        std::map<datapoint<bool> *, double> _datapoint_ptrs_to_frac_temp(_datapoint_ptrs_to_frac);
        for (auto positive_ptr : positive_ptrs) {
            _datapoint_ptrs_to_frac_temp[positive_ptr] = 1.0;
        }
        for (auto negative_ptr : negative_ptrs) {
            _datapoint_ptrs_to_frac_temp[negative_ptr] = 0.0;
        }
        for (auto & it : _datapoint_ptrs_to_frac_temp) {
            _sum_of_datapoint_ptrs_to_frac[it.first] += it.second;
        }
    }
    // Divide _sum_of_datapoint_ptrs_to_frac by 5 and store it to _datapoint_ptrs_to_frac
    for (auto it = _sum_of_datapoint_ptrs_to_frac.begin(); it != _sum_of_datapoint_ptrs_to_frac.end(); ++it) {
        _sum_of_datapoint_ptrs_to_frac[it->first] = it->second / numberOfCompleteHornAssignments;
    }

    // Update _datapoint_ptrs_to_frac to _sum_of_datapoint_ptrs_to_frac
    _datapoint_ptrs_to_frac = _sum_of_datapoint_ptrs_to_frac;
}

std::unique_ptr<abstract_job> complex_job_manager::next_job() {
    if (_is_first_split) {
        srand(time(NULL));
        auto sl = _slices.front();
        _slices.pop_front();

        // Check if data points have exactly one categorical attribute
        if (_datapoint_ptrs[sl._left_index]->_categorical_data.size() != 1) {
            throw std::runtime_error("Learner expects exactly one categorical attribute");
        }
        _is_first_split = false;

        return std::unique_ptr<abstract_job>{std::make_unique<categorical_split_job>(sl, 0)};
    } else {
        if (_node_selection_criterion == BFS || _node_selection_criterion == RANDOM ||
            _node_selection_criterion == DFS) {
            auto slice_index =
                _node_selection_criterion == BFS
                ? 0
                : _node_selection_criterion == DFS ? _slices.size() - 1 : rand() % _slices.size();
            auto it = _slices.begin();
            advance(it, slice_index);
            auto sl = *it;
            _slices.erase(it);

            //
            // Determine what needs to be done (split or create leaf)
            //
            auto label = false; // label is unimportant (if is_leaf() returns false)
            auto positive_ptrs = std::unordered_set<datapoint<bool> *>();
            auto negative_ptrs = std::unordered_set<datapoint<bool> *>();

            auto can_be_turned_into_leaf = is_leaf(sl, label, positive_ptrs, negative_ptrs);

            // Slice can be turned into a leaf node
            if (can_be_turned_into_leaf) {
                return std::unique_ptr<abstract_job>{std::make_unique<leaf_creation_job>(
                    sl, label, std::move(positive_ptrs), std::move(negative_ptrs))};
            }
                // Slice needs to be split
            else {
                if (_entropy_computation_criterion == DEFAULT_ENTROPY ||
                    _entropy_computation_criterion == PENALTY) {
                    return find_best_split(sl);
                } else if (_entropy_computation_criterion == HORN_ASSIGNMENTS) {

                    // Clear _datapoint_ptrs_to_frac from a previous iteration
                    _datapoint_ptrs_to_frac.clear();

                    // Initialize the _datapoint_ptrs_to_frac map using the classified points in _datapoint_ptrs
                    initialize_datapoint_ptrs_to_frac();

                    if (!unclassified_points_present(_datapoint_ptrs, sl._left_index, sl._right_index)) {
                        return find_best_split(sl);
                    } else {
                        // Update _datapoint_ptrs_to_frac with randomly selected complete horn assignments
                        update_datapoint_ptrs_to_frac_with_complete_horn_assignments();
                        return find_best_split(sl);
                    }
                } else {
                    // Control never reaches here!
                    assert(false);
                }
            }
        } else if (_node_selection_criterion == MAX_ENTROPY || _node_selection_criterion == MAX_WEIGHTED_ENTROPY) {
            if (_entropy_computation_criterion == HORN_ASSIGNMENTS) {
                _datapoint_ptrs_to_frac.clear();
                initialize_datapoint_ptrs_to_frac();
                update_datapoint_ptrs_to_frac_with_complete_horn_assignments();
            }

            float max_entropy = 0.0;
            unsigned int max_entropy_slice_index = 0;
            unsigned int cur_index = 0;
            for (auto & _slice : _slices) {
                auto entropy_val = _node_selection_criterion == MAX_ENTROPY
                                   ? entropy(_datapoint_ptrs, _slice._left_index, _slice._right_index)
                                   : weighted_entropy(_datapoint_ptrs, _slice._left_index, _slice._right_index);
                if (entropy_val > max_entropy) {
                    max_entropy = entropy_val;
                    max_entropy_slice_index = cur_index;
                }
                cur_index++;
            }

            assert(max_entropy_slice_index >= 0 && max_entropy_slice_index < _slices.size());
            auto it = _slices.begin();
            advance(it, max_entropy_slice_index);

            auto sl = *it;
            _slices.erase(it);

            //
            // Determine what needs to be done (split or create leaf)
            //
            auto label = false; // label is unimportant (if is_leaf() returns false)
            auto positive_ptrs = std::unordered_set<datapoint<bool> *>();
            auto negative_ptrs = std::unordered_set<datapoint<bool> *>();
            auto can_be_turned_into_leaf = is_leaf(sl, label, positive_ptrs, negative_ptrs);

            // Slice can be turned into a leaf node
            if (can_be_turned_into_leaf) {
                return std::unique_ptr<abstract_job>{std::make_unique<leaf_creation_job>(
                    sl, label, std::move(positive_ptrs), std::move(negative_ptrs))};
            }
                // Slice needs to be split
            else {
                return find_best_split(sl);
            }

        } else if (_node_selection_criterion == MIN_ENTROPY || _node_selection_criterion == MIN_WEIGHTED_ENTROPY) {
            if (_entropy_computation_criterion == HORN_ASSIGNMENTS) {
                _datapoint_ptrs_to_frac.clear();
                initialize_datapoint_ptrs_to_frac();
                update_datapoint_ptrs_to_frac_with_complete_horn_assignments();
            }

            float min_entropy = _node_selection_criterion == MIN_ENTROPY ? 1.0 : 100000.0;
            unsigned int min_entropy_slice_index = 0;
            unsigned int cur_index = 0;
            for (auto & _slice : _slices) {
                auto entropy_val = _node_selection_criterion == MIN_ENTROPY
                                   ? entropy(_datapoint_ptrs, _slice._left_index, _slice._right_index)
                                   : weighted_entropy(_datapoint_ptrs, _slice._left_index, _slice._right_index);
                if (entropy_val < min_entropy) {
                    min_entropy = entropy_val;
                    min_entropy_slice_index = cur_index;
                }
                cur_index++;
            }

            assert(min_entropy_slice_index >= 0 && min_entropy_slice_index < _slices.size());
            auto it = _slices.begin();
            advance(it, min_entropy_slice_index);
            auto sl = *it;
            _slices.erase(it);

            //
            // Determine what needs to be done (split or create leaf)
            //
            auto label = false; // label is unimportant (if is_leaf() returns false)
            auto positive_ptrs = std::unordered_set<datapoint<bool> *>();
            auto negative_ptrs = std::unordered_set<datapoint<bool> *>();
            auto can_be_turned_into_leaf = is_leaf(sl, label, positive_ptrs, negative_ptrs);

            // Slice can be turned into a leaf node
            if (can_be_turned_into_leaf) {
                return std::unique_ptr<abstract_job>{std::make_unique<leaf_creation_job>(
                    sl, label, std::move(positive_ptrs), std::move(negative_ptrs))};
            }
                // Slice needs to be split
            else {
                return find_best_split(sl);
            }

        } else {
            // Control never reaches here!
            assert(false);
        }
    }
}

double complex_job_manager::entropy(
    const std::vector<datapoint<bool> *> &datapoint_ptrs, std::size_t left_index, std::size_t right_index) {
    if (_entropy_computation_criterion == HORN_ASSIGNMENTS) {
        double count_f = 0.0;
        double count_t = 0.0;

        for (std::size_t i = left_index; i <= right_index; ++i) {
            // assert that datapoint_ptrs[i] is present in the map _datapoint_ptrs_to_frac
            auto it = _datapoint_ptrs_to_frac.find(datapoint_ptrs[i]);
            assert(it != _datapoint_ptrs_to_frac.end());
            count_t += it->second;
        }

        double sum = right_index - left_index + 1;
        count_f = sum - count_t;

        // std::cout << "sum=" << sum << std::endl;
        double p_t = ((double)(count_t) / sum);
        // std::cout << "p_t=" << p_t << std::endl;
        double p_f = ((double)(count_f) / sum);
        // std::cout << "p_f=" << p_f << std::endl;

        double entropy_t = count_t == 0.0 ? 0 : p_t * log2(p_t);
        double entropy_f = count_f == 0.0 ? 0 : p_f * log2(p_f);

        return -(entropy_t + entropy_f);

    } else if (_entropy_computation_criterion == DEFAULT_ENTROPY || _entropy_computation_criterion == PENALTY) {
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
    // _entropy_computation_criterion should be one of the two implemented criterions
    // Control should never reach here!
    assert(false);
    return 0.0;
}

double complex_job_manager::weighted_entropy(
    const std::vector<datapoint<bool> *> &datapoint_ptrs, std::size_t left_index, std::size_t right_index) {
    if (_entropy_computation_criterion == HORN_ASSIGNMENTS) {
        return entropy(datapoint_ptrs, left_index, right_index) * (right_index - left_index + 1);
    } else if (_entropy_computation_criterion == DEFAULT_ENTROPY || _entropy_computation_criterion == PENALTY) {
        return entropy(datapoint_ptrs, left_index, right_index) *
               num_classified_points(_datapoint_ptrs, left_index, right_index);
    }
    // _entropy_computation_criterion should be one of the two implemented criteria
    // Control should never reach here!
    assert(false);
    return 0.0;
}

unsigned int complex_job_manager::num_classified_points(
    const std::vector<datapoint<bool> *> &datapoint_ptrs, std::size_t left_index, std::size_t right_index) {
    if (_entropy_computation_criterion == DEFAULT_ENTROPY || _entropy_computation_criterion == PENALTY) {
        unsigned int count = 0;

        for (std::size_t i = left_index; i <= right_index; ++i) {
            if (datapoint_ptrs[i]->_is_classified) {
                count++;
            }
        }
        return count;
    } else if (_entropy_computation_criterion == HORN_ASSIGNMENTS) {
        return right_index - left_index + 1;
    }
    assert(false);
    return 0;
}

std::unique_ptr<abstract_job> complex_job_manager::find_best_split(const slice &sl) {
    assert(sl._left_index <= sl._right_index && sl._right_index < _datapoint_ptrs.size());

    // 0) Initialize variables
    bool int_split_possible = false;
    bool non_zero_intrinsic_value_possible = false;
    double best_int_gain_ratio = -1000000;
    std::size_t best_int_attribute = 0;
    int best_int_threshold = 0;
    double best_int_gain_ratio_4_zero_iv = -1000000;
    std::size_t best_int_attribute_4_zero_iv = 0;
    int best_int_threshold_4_zero_iv = 0;

    // Variables to track the split if conjunctive splits are preferred
    bool conj_int_split_possible = false;
    bool conj_non_zero_intrinsic_value_possible = false;
    double best_conj_int_gain_ratio = -1000000;
    std::size_t best_conj_int_attribute = 0;
    int best_conj_int_threshold = 0;
    double best_conj_int_gain_ratio_4_zero_iv = -1000000;
    std::size_t best_conj_int_attribute_4_zero_iv = 0;
    int best_conj_int_threshold_4_zero_iv = 0;

    bool cat_split_possible = false;
    double best_cat_gain_ratio = -1000000;
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
        std::sort(
            _datapoint_ptrs.begin() + sl._left_index, _datapoint_ptrs.begin() + sl._right_index + 1, comparer);

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
                info_gain = entropy(_datapoint_ptrs, sl._left_index, sl._right_index) -
                            total_weighted_entropy /
                            num_classified_points(_datapoint_ptrs, sl._left_index, sl._right_index);
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

    //
    // Process integer attributes
    //
    for (std::size_t attribute = 0; attribute < _datapoint_ptrs[sl._left_index]->_int_data.size(); ++attribute) {
        int tries = 0;
        double best_int_entropy_for_given_attribute = 1000000;
        bool int_split_possible_for_given_attribute = false;
        int best_int_split_index_for_given_attribute = 0;
        double best_intrinsic_value_for_given_attribute = 0;

        double best_conj_int_entropy_for_given_attribute = 1000000;
        bool conj_int_split_possible_for_given_attribute = false;
        int best_conj_int_split_index_for_given_attribute = 0;
        double best_conj_intrinsic_value_for_given_attribute = 0;

        // 1) Sort according to int attribute
        auto comparer = [attribute](const datapoint<bool> *const a, const datapoint<bool> *const b) {
            return a->_int_data[attribute] < b->_int_data[attribute];
        };
        std::sort(
            _datapoint_ptrs.begin() + sl._left_index, _datapoint_ptrs.begin() + sl._right_index + 1, comparer);

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

                // if cuts have been thresholded, check that a split at the current value of the numerical attribute
                // is allowed
                if (!_are_numerical_cuts_thresholded ||
                    ((-1 * _threshold <= _datapoint_ptrs[cur]->_int_data[attribute]) &&
                     (_datapoint_ptrs[cur]->_int_data[attribute] <= _threshold))) {
                    auto weighted_entropy_left = weighted_entropy(_datapoint_ptrs, sl._left_index, cur);
                    auto weighted_entropy_right = weighted_entropy(_datapoint_ptrs, cur + 1, sl._right_index);
                    auto total_weighted_entropy = weighted_entropy_left + weighted_entropy_right;
                    double total_entropy;
                    if (num_classified_points(_datapoint_ptrs, sl._left_index, sl._right_index) == 0.0) {
                        total_entropy = 0.0;
                    } else {
                        total_entropy =
                            total_weighted_entropy /
                            (double)num_classified_points(_datapoint_ptrs, sl._left_index, sl._right_index);
                    }

                    // Add a penalty based on the number of implications in the horn constraints that are cut by the
                    // current split.
                    if (_entropy_computation_criterion == PENALTY) {
                        // number of implications in the horn constraints cut by the current split.
                        int left2right = 0;
                        int right2left = 0;
                        penalty(sl, sl._left_index, cur, sl._right_index, &left2right, &right2left);
                        double nleft = num_points_with_classification(_datapoint_ptrs, sl._left_index, cur, false);
                        double pleft = num_points_with_classification(_datapoint_ptrs, sl._left_index, cur, true);
                        double nright =
                            num_points_with_classification(_datapoint_ptrs, cur + 1, sl._right_index, false);
                        double pright =
                            num_points_with_classification(_datapoint_ptrs, cur + 1, sl._right_index, true);
                        double total_classified_points = nleft + pleft + nright + pright;

                        nleft = nleft == 0 ? 0 : nleft / (nleft + pleft);
                        pleft = pleft == 0 ? 0 : pleft / (nleft + pleft);
                        nright = nright == 0 ? 0 : nright / (nright + pright);
                        pright = pright == 0 ? 0 : pright / (nright + pright);

                        double penaltyVal = (1 - nleft * pright) * left2right + (1 - nright * pleft) * right2left;
                        penaltyVal = 2 * penaltyVal / (2 * (left2right + right2left) + total_classified_points);
                        total_entropy += penaltyVal;
                    }

                    // If the learner prefers conjunctive splits
                    if (_conjunctive_setting == PREFERENCEFORCONJUNCTS) {
                        // Check if a conjunctive split is possible
                        if (!positive_points_present(_datapoint_ptrs, sl._left_index, cur) ||
                            !positive_points_present(_datapoint_ptrs, cur + 1, sl._right_index)) {
                            // One of the sub node consists purely of negative or unclassified points.
                            // Consider this as a prospective candidate for a conjunctive split

                            if (!conj_int_split_possible_for_given_attribute ||
                                total_entropy < best_conj_int_entropy_for_given_attribute) {
                                conj_int_split_possible_for_given_attribute = true;

                                best_conj_int_entropy_for_given_attribute = total_entropy;
                                best_conj_int_split_index_for_given_attribute = cur;

                                // computation of the intrinsic value of the attribute
                                double n1 = 1.0 * num_classified_points(_datapoint_ptrs, sl._left_index, cur);
                                double n2 = 1.0 * num_classified_points(_datapoint_ptrs, cur + 1, sl._right_index);
                                double n = n1 + n2;
                                best_conj_intrinsic_value_for_given_attribute =
                                    (n1 == 0.0 ? 0.0 : -1.0 * (n1 / n) * log2(n1 / n)) +
                                    (n2 == 0.0 ? 0.0 : -1.0 * (n2 / n) * log2(n2 / n));
                            }
                        }
                    }

                    if (!int_split_possible_for_given_attribute ||
                        total_entropy < best_int_entropy_for_given_attribute) {
                        int_split_possible_for_given_attribute = true;

                        best_int_entropy_for_given_attribute = total_entropy;
                        best_int_split_index_for_given_attribute = cur;

                        // computation of the intrinsic value of the attribute
                        double n1 = 1.0 * num_classified_points(_datapoint_ptrs, sl._left_index, cur);
                        double n2 = 1.0 * num_classified_points(_datapoint_ptrs, cur + 1, sl._right_index);
                        double n = n1 + n2;
                        best_intrinsic_value_for_given_attribute =
                            (n1 == 0.0 ? 0.0 : -1.0 * (n1 / n) * log2(n1 / n)) +
                            (n2 == 0.0 ? 0.0 : -1.0 * (n2 / n) * log2(n2 / n));
                    }
                }

                ++cur;
            }
        }

        if (_conjunctive_setting == PREFERENCEFORCONJUNCTS && conj_int_split_possible_for_given_attribute) {
            // We have found the best split threshold for the given attribute
            // Now compute the information gain to optimize across different attributes
            double best_info_gain_for_attribute;
            best_info_gain_for_attribute = entropy(_datapoint_ptrs, sl._left_index, sl._right_index) -
                                           best_conj_int_entropy_for_given_attribute;

            double interval =
                (_datapoint_ptrs[sl._right_index]->_int_data[attribute] -
                 _datapoint_ptrs[sl._left_index]->_int_data[attribute]) /
                (_datapoint_ptrs[best_conj_int_split_index_for_given_attribute + 1]->_int_data[attribute] -
                 _datapoint_ptrs[best_conj_int_split_index_for_given_attribute]->_int_data[attribute]);

            assert(num_classified_points(_datapoint_ptrs, sl._left_index, sl._right_index) > 0);
            double threshCost = (interval < (double)tries ? log2(interval) : log2(tries)) /
                                num_classified_points(_datapoint_ptrs, sl._left_index, sl._right_index);

            best_info_gain_for_attribute -= threshCost;
            if (best_conj_intrinsic_value_for_given_attribute > 0.0) {
                conj_non_zero_intrinsic_value_possible = true;
                double best_gain_ratio_for_given_attribute =
                    best_info_gain_for_attribute / best_conj_intrinsic_value_for_given_attribute;

                if (!conj_int_split_possible || (best_gain_ratio_for_given_attribute > best_conj_int_gain_ratio) ||
                    (best_gain_ratio_for_given_attribute == best_conj_int_gain_ratio &&
                     abs(_datapoint_ptrs[best_conj_int_split_index_for_given_attribute]->_int_data[attribute]) <
                     abs(best_conj_int_threshold))) {
                    // if this is the first attribute for which a split is possible then
                    // initialize all variables: best_int_gain_ratio, best_int_attribute, best_int_threshold
                    conj_int_split_possible = true;
                    best_conj_int_gain_ratio = best_gain_ratio_for_given_attribute;
                    best_conj_int_attribute = attribute;
                    best_conj_int_threshold =
                        _datapoint_ptrs[best_conj_int_split_index_for_given_attribute]->_int_data[attribute];
                }
            } else {
                if (!conj_non_zero_intrinsic_value_possible) {
                    double best_gain_ratio_for_given_attribute = best_info_gain_for_attribute;

                    if (!conj_int_split_possible ||
                        (best_gain_ratio_for_given_attribute > best_conj_int_gain_ratio_4_zero_iv) ||
                        (best_gain_ratio_for_given_attribute == best_conj_int_gain_ratio_4_zero_iv &&
                         abs(_datapoint_ptrs[best_conj_int_split_index_for_given_attribute]->_int_data[attribute]) <
                         abs(best_conj_int_threshold_4_zero_iv))) {
                        // if this is the first attribute for which a split is possible then
                        // initialize all variables: best_int_gain_ratio, best_int_attribute, best_int_threshold
                        conj_int_split_possible = true;
                        best_conj_int_gain_ratio_4_zero_iv = best_gain_ratio_for_given_attribute;
                        best_conj_int_attribute_4_zero_iv = attribute;
                        best_conj_int_threshold_4_zero_iv =
                            _datapoint_ptrs[best_conj_int_split_index_for_given_attribute]->_int_data[attribute];
                    }
                }
            }
        }

        if (int_split_possible_for_given_attribute) {
            // We have found the best split threshold for the given attribute
            // Now compute the information gain to optimize across different attributes
            double best_info_gain_for_attribute;
            best_info_gain_for_attribute =
                entropy(_datapoint_ptrs, sl._left_index, sl._right_index) - best_int_entropy_for_given_attribute;

            double interval = (_datapoint_ptrs[sl._right_index]->_int_data[attribute] -
                               _datapoint_ptrs[sl._left_index]->_int_data[attribute]) /
                              (_datapoint_ptrs[best_int_split_index_for_given_attribute + 1]->_int_data[attribute] -
                               _datapoint_ptrs[best_int_split_index_for_given_attribute]->_int_data[attribute]);

            assert(num_classified_points(_datapoint_ptrs, sl._left_index, sl._right_index) > 0);
            double threshCost = (interval < (double)tries ? log2(interval) : log2(tries)) /
                                num_classified_points(_datapoint_ptrs, sl._left_index, sl._right_index);

            best_info_gain_for_attribute -= threshCost;
            if (best_intrinsic_value_for_given_attribute > 0.0) {
                non_zero_intrinsic_value_possible = true;
                double best_gain_ratio_for_given_attribute =
                    best_info_gain_for_attribute / best_intrinsic_value_for_given_attribute;

                if (!int_split_possible || (best_gain_ratio_for_given_attribute > best_int_gain_ratio) ||
                    (best_gain_ratio_for_given_attribute == best_int_gain_ratio &&
                     abs(_datapoint_ptrs[best_int_split_index_for_given_attribute]->_int_data[attribute]) <
                     abs(best_int_threshold))) {
                    // if this is the first attribute for which a split is possible then
                    // initialize all variables: best_int_gain_ratio, best_int_attribute, best_int_threshold
                    int_split_possible = true;
                    best_int_gain_ratio = best_gain_ratio_for_given_attribute;
                    best_int_attribute = attribute;
                    best_int_threshold =
                        _datapoint_ptrs[best_int_split_index_for_given_attribute]->_int_data[attribute];
                }
            } else {
                if (!non_zero_intrinsic_value_possible) {
                    double best_gain_ratio_for_given_attribute = best_info_gain_for_attribute;

                    if (!int_split_possible ||
                        (best_gain_ratio_for_given_attribute > best_int_gain_ratio_4_zero_iv) ||
                        (best_gain_ratio_for_given_attribute == best_int_gain_ratio_4_zero_iv &&
                         abs(_datapoint_ptrs[best_int_split_index_for_given_attribute]->_int_data[attribute]) <
                         abs(best_int_threshold_4_zero_iv))) {
                        // if this is the first attribute for which a split is possible then
                        // initialize all variables: best_int_gain_ratio, best_int_attribute, best_int_threshold
                        int_split_possible = true;
                        best_int_gain_ratio_4_zero_iv = best_gain_ratio_for_given_attribute;
                        best_int_attribute_4_zero_iv = attribute;
                        best_int_threshold_4_zero_iv =
                            _datapoint_ptrs[best_int_split_index_for_given_attribute]->_int_data[attribute];
                    }
                }
            }
        }
    }

    //
    // Return best split
    //
    if (!int_split_possible && !cat_split_possible) {
        assert(!conj_int_split_possible);
        throw split_not_possible_error("No split possible!");
    } else if (int_split_possible && !cat_split_possible) {
        if (_conjunctive_setting == PREFERENCEFORCONJUNCTS && conj_int_split_possible) {
            if (!conj_non_zero_intrinsic_value_possible) {
                best_conj_int_gain_ratio = best_conj_int_gain_ratio_4_zero_iv;
                best_conj_int_attribute = best_conj_int_attribute_4_zero_iv;
                best_conj_int_threshold = best_conj_int_threshold_4_zero_iv;
            }
            return std::unique_ptr<abstract_job>{
                std::make_unique<int_split_job>(sl, best_conj_int_attribute, best_conj_int_threshold)};
        } else {
            if (!non_zero_intrinsic_value_possible) {
                best_int_gain_ratio = best_int_gain_ratio_4_zero_iv;
                best_int_attribute = best_int_attribute_4_zero_iv;
                best_int_threshold = best_int_threshold_4_zero_iv;
            }
            return std::unique_ptr<abstract_job>{
                std::make_unique<int_split_job>(sl, best_int_attribute, best_int_threshold)};
        }
    } else if (!int_split_possible && cat_split_possible) {
        assert(!conj_int_split_possible);
        return std::unique_ptr<abstract_job>{std::make_unique<categorical_split_job>(sl, best_cat_attribute)};
    } else {
        // If conjunctive splits are preferred, overwrite the best int split variabls with those corresponding to
        // conjunctive splits.
        if (_conjunctive_setting == PREFERENCEFORCONJUNCTS && conj_int_split_possible) {
            if (!conj_non_zero_intrinsic_value_possible) {
                best_int_gain_ratio = best_conj_int_gain_ratio_4_zero_iv;
                best_int_attribute = best_conj_int_attribute_4_zero_iv;
                best_int_threshold = best_conj_int_threshold_4_zero_iv;
            } else {
                best_int_gain_ratio = best_conj_int_gain_ratio;
                best_int_attribute = best_conj_int_attribute;
                best_int_threshold = best_conj_int_threshold;
            }
        } else {
            if (!non_zero_intrinsic_value_possible) {
                best_int_gain_ratio = best_int_gain_ratio_4_zero_iv;
                best_int_attribute = best_int_attribute_4_zero_iv;
                best_int_threshold = best_int_threshold_4_zero_iv;
            }
        }
        if (best_int_gain_ratio <= best_cat_gain_ratio) {
            return std::unique_ptr<abstract_job>{
                std::make_unique<int_split_job>(sl, best_int_attribute, best_int_threshold)};
        } else {
            return std::unique_ptr<abstract_job>{std::make_unique<categorical_split_job>(sl, best_cat_attribute)};
        }
    }
}

void complex_job_manager::penalty(
    const slice &sl,
    std::size_t left_index,
    std::size_t cur_index,
    std::size_t right_index,
    int *left2right,
    int *right2left) {
    int _left2right = 0;
    int _right2left = 0;
    for (const auto &horn_clause : _horn_constraints) {
        enum Position { out_of_scope, left, right };
        Position conclusion = out_of_scope;
        int num_premise_left = 0;
        int num_premise_right = 0;

        // for i ranging from left_index to cur, loop over premises and conclusion
        for (std::size_t i = left_index; i <= cur_index; ++i) {
            for (const auto dp : horn_clause._premises) {
                if (dp == _datapoint_ptrs[i] && !_datapoint_ptrs[i]->_is_classified) {
                    num_premise_left++;
                }
            }
            if (_datapoint_ptrs[i] == horn_clause._conclusion && !_datapoint_ptrs[i]->_is_classified) {
                conclusion = left;
            }
        }

        // for i ranging from cur+1 to right_index, loop over premises and conclusion
        for (std::size_t i = cur_index + 1; i <= right_index; ++i) {
            for (const auto dp : horn_clause._premises) {
                if (dp == _datapoint_ptrs[i] && !_datapoint_ptrs[i]->_is_classified) {
                    num_premise_right++;
                }
            }
            if (_datapoint_ptrs[i] == horn_clause._conclusion && !_datapoint_ptrs[i]->_is_classified) {
                conclusion = right;
            }
        }
        if (conclusion == left) {
            _right2left += num_premise_right;
        }
        if (conclusion == right) {
            _left2right += num_premise_left;
        }
    }
    *right2left = _right2left;
    *left2right = _left2right;
}
