#include "util.h"
#include "complex_job_manager.h"

using namespace horn_verification;

void complex_job_manager::initialize_datapoint_ptrs_to_frac() {
    for (auto & _datapoint_ptr : _datapoint_ptrs) {
        if (_datapoint_ptr->_is_classified) {
            _datapoint_ptrs_to_frac[_datapoint_ptr] = _datapoint_ptr->_classification ? 1.0 : 0.0;
        }
    }
}

void complex_job_manager::update_datapoint_ptrs_to_frac_with_complete_horn_assignments() {
    std::map<datapoint<bool> *, double> _sum_of_datapoint_ptrs_to_frac;

    // Create a list of _datapoint_ptrs which are unclassified
    std::unordered_set<datapoint<bool> *> _unclassified_datapoint_ptrs_stable;
    for (auto & _datapoint_ptr : _datapoint_ptrs) {
        if (!_datapoint_ptr->_is_classified) {
            _unclassified_datapoint_ptrs_stable.emplace(_datapoint_ptr);
        }
        _sum_of_datapoint_ptrs_to_frac[_datapoint_ptr] = 0.0;
    }

    // Now generate complete horn assignments by randomly picking a datapoint from _unclassified_datapoint_ptrs and
    // assigning it True/False followed by label propagation.
    constexpr int numberOfCompleteHornAssignments = 3;
    for (int i = 0; i < numberOfCompleteHornAssignments; i++) {
        auto _unclassified_datapoint_ptrs_temp(_unclassified_datapoint_ptrs_stable);
        std::unordered_set<datapoint<bool> *> positive_ptrs;
        std::unordered_set<datapoint<bool> *> negative_ptrs;

        while (!_unclassified_datapoint_ptrs_temp.empty()) {
            unsigned int itemToAssignClassification = rand() % _unclassified_datapoint_ptrs_temp.size();
            auto item = *std::next(std::begin(_unclassified_datapoint_ptrs_temp), itemToAssignClassification);
            (rand() % 2 == 0 ? positive_ptrs : negative_ptrs).insert(item);

            auto ok = _horn_solver.solve(_datapoint_ptrs, _horn_constraints, positive_ptrs, negative_ptrs);
            assert(ok);

            // Remove the items present in positive_ptrs and negative_ptrs from _unclassified_datapoint_ptrs_temp
            for (auto it = _unclassified_datapoint_ptrs_temp.begin(); it != _unclassified_datapoint_ptrs_temp.end();) {
                if (positive_ptrs.find(*it) != positive_ptrs.end() || negative_ptrs.find(*it) != negative_ptrs.end()) {
                    it = _unclassified_datapoint_ptrs_temp.erase(it);
                } else {
                    ++it;
                }
            }
        }

        for (auto positive_ptr : positive_ptrs) {
            _sum_of_datapoint_ptrs_to_frac[positive_ptr] += 1.0;
        }
    }
    // Divide _sum_of_datapoint_ptrs_to_frac by 5 and store it to _datapoint_ptrs_to_frac
    for (auto &value : _sum_of_datapoint_ptrs_to_frac) {
        value.second /= numberOfCompleteHornAssignments;
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
    } 

    size_t slice_index;
    bool is_weighted = false;
    switch (_node_selection_criterion) {
        case NodeSelection::BFS:
            slice_index = 0; break;

        case NodeSelection::RANDOM:
            slice_index = rand() % _slices.size(); break;

        case NodeSelection::DFS:
            slice_index = _slices.size() - 1; break;

        case NodeSelection::MAX_WEIGHTED_ENTROPY:
        case NodeSelection::MIN_WEIGHTED_ENTROPY:
            is_weighted = true;
        case NodeSelection::MAX_ENTROPY:
        case NodeSelection::MIN_ENTROPY:
            float best_entropy;
            bool (*cmp)(double, double);
            if (_node_selection_criterion == NodeSelection::MIN_ENTROPY ||
                _node_selection_criterion == NodeSelection::MIN_WEIGHTED_ENTROPY) {
                best_entropy = _node_selection_criterion == NodeSelection::MIN_ENTROPY ? 1.0 : 100000.0;
                cmp = &less<double>;
            } else {
                best_entropy = 0;
                cmp = &greater<double>;
            }
            size_t cur_index = 0;
            for (auto & _slice : _slices) {
                auto entropy_val = is_weighted
                                    ? weighted_entropy(_datapoint_ptrs, _slice._left_index, _slice._right_index)
                                    : entropy(_datapoint_ptrs, _slice._left_index, _slice._right_index);
                if (cmp(entropy_val, best_entropy)) {
                    best_entropy = entropy_val;
                    slice_index = cur_index;
                }
                ++cur_index;
            }
            break;
        default:
            assert(false);
    }
    assert(slice_index >= 0 && slice_index < _slices.size());

    auto it = _slices.begin();
    advance(it, slice_index);
    auto sl = *it;
    _slices.erase(it);

    if (_entropy_computation_criterion == EntropyComputation::HORN_ASSIGNMENTS) {
        // Clear _datapoint_ptrs_to_frac from a previous iteration
        _datapoint_ptrs_to_frac.clear();

        // Initialize the _datapoint_ptrs_to_frac map using the classified points in _datapoint_ptrs
        initialize_datapoint_ptrs_to_frac();

        if (unclassified_points_present(_datapoint_ptrs, sl._left_index, sl._right_index)) {
            // Update _datapoint_ptrs_to_frac with randomly selected complete horn assignments
            update_datapoint_ptrs_to_frac_with_complete_horn_assignments();
        }
    }

    bool label = false; // label is unimportant (if is_leaf() returns false)
    std::unordered_set<datapoint<bool> *> positive_ptrs, negative_ptrs;

    auto can_be_turned_into_leaf = is_leaf(sl, label, positive_ptrs, negative_ptrs);

    // Slice can be turned into a leaf node
    if (can_be_turned_into_leaf) {
        return std::make_unique<leaf_creation_job>(
            sl, label, std::move(positive_ptrs), std::move(negative_ptrs));
    }
    // Slice needs to be split
    else {
        return find_best_split(sl);
    }
}

double complex_job_manager::entropy(
    const std::vector<datapoint<bool> *> &datapoint_ptrs, std::size_t left_index, std::size_t right_index) {
    if (_entropy_computation_criterion == EntropyComputation::HORN_ASSIGNMENTS) {
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

        // TODO: the following is the same as in the default case
        // std::cout << "sum=" << sum << std::endl;
        double p_t = ((double)(count_t) / sum);
        // std::cout << "p_t=" << p_t << std::endl;
        double p_f = ((double)(count_f) / sum);
        // std::cout << "p_f=" << p_f << std::endl;

        double entropy_t = count_t == 0.0 ? 0 : p_t * log2(p_t);
        double entropy_f = count_f == 0.0 ? 0 : p_f * log2(p_f);

        return -(entropy_t + entropy_f);

    } else if (_entropy_computation_criterion == EntropyComputation::DEFAULT_ENTROPY ||
        _entropy_computation_criterion == EntropyComputation::PENALTY) {
        return simple_job_manager::entropy(datapoint_ptrs, left_index, right_index);
    }
    // _entropy_computation_criterion should be one of the two implemented criterions
    // Control should never reach here!
    assert(false);
    return 0.0;
}

unsigned int complex_job_manager::num_classified_points(
    const std::vector<datapoint<bool> *> &datapoint_ptrs, std::size_t left_index, std::size_t right_index) {
    if (_entropy_computation_criterion == EntropyComputation::DEFAULT_ENTROPY ||
        _entropy_computation_criterion == EntropyComputation::PENALTY) {
        return simple_job_manager::num_classified_points(datapoint_ptrs, left_index, right_index);
    } else if (_entropy_computation_criterion == EntropyComputation::HORN_ASSIGNMENTS) {
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
                    if (_entropy_computation_criterion == EntropyComputation::PENALTY) {
                        // number of implications in the horn constraints cut by the current split.
                        int left2right, right2left;
                        std::tie(left2right, right2left) = penalty(sl._left_index, cur, sl._right_index);
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
                    if (_conjunctive_setting == ConjunctiveSetting::PREFERENCEFORCONJUNCTS) {
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

        if (_conjunctive_setting == ConjunctiveSetting::PREFERENCEFORCONJUNCTS &&
            conj_int_split_possible_for_given_attribute) {
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
        if (_conjunctive_setting == ConjunctiveSetting::PREFERENCEFORCONJUNCTS &&
            conj_int_split_possible) {
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
        if (_conjunctive_setting == ConjunctiveSetting::PREFERENCEFORCONJUNCTS &&
            conj_int_split_possible) {
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

std::pair<int, int> complex_job_manager::penalty(
    std::size_t left_index,
    std::size_t cur_index,
    std::size_t right_index) {
    int _left2right = 0;
    int _right2left = 0;
    for (const auto &horn_clause : _horn_constraints) {
        enum class Position { out_of_scope, left, right };
        Position conclusion = Position::out_of_scope;
        int num_premise_left = 0;
        int num_premise_right = 0;

        // OPTIMIZE: this is an O(nm) loop, but it really should be O(n+m) or at least O(n log n+m log m)
        // for i ranging from left_index to cur, loop over premises and conclusion
        auto left_overlap = count_overlap(horn_clause, left_index, cur_index);
        auto right_overlap = count_overlap(horn_clause, cur_index + 1, right_index);
        if (left_overlap.second) {
            _right2left += right_overlap.first;
        } else if (right_overlap.second) {
            _left2right += left_overlap.first;
        }
    }
    return { _left2right, _right2left };
}

std::pair<size_t, bool> complex_job_manager::count_overlap(
    horn_constraint<bool> horn_clause,
    std::size_t start_index,
    std::size_t end_index) {
    size_t num_premises = 0;
    bool has_conclusion = false;

    for (std::size_t i = start_index; i <= end_index; ++i) {
        for (const auto dp : horn_clause._premises) {
            if (dp == _datapoint_ptrs[i] && !_datapoint_ptrs[i]->_is_classified) {
                num_premises++;
            }
        }
        if (_datapoint_ptrs[i] == horn_clause._conclusion && !_datapoint_ptrs[i]->_is_classified) {
            has_conclusion = true;
        }
    }

    return { num_premises, has_conclusion };
}